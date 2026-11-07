//! # MinHash Deduplication Utilities
//!
//! This module provides shared utilities for MinHash-based document deduplication,
//! supporting both in-memory and disk-based processing strategies.
//!
//! ## Overview
//!
//! The deduplication pipeline consists of four main stages:
//! 1. **File Mapping**: Map file paths to indices for efficient storage
//! 2. **Hashing**: Compute MinHash signatures for all documents
//! 3. **Edge Gathering**: Collect duplicate candidates with matching signatures
//! 4. **Union-Find**: Merge duplicates and generate cleaning metadata
//!
//! ## Parallelism Strategy
//!
//! - **File Mapping**: Global (single-node local or Python tool for distributed)
//! - **Hashing**: Parallel across file path chunks
//! - **Edge Gathering**: Parallel across band IDs
//! - **Union-Find**: Must run globally (no multi-node parallelism)
//! - **Cleaning**: Parallel across file path chunks

use crate::minhash_config::Config;
use crate::storage::GenWriter;
use crate::storage::{compute_sig_size, to_byte_size, IntValueEnum, SignatureWriter};
use crate::uf_rush2::{parent as uf_parent, UFRush};
use crate::utils::json_set;
use ahash::RandomState;
use anyhow::{Error, Result};
use dashmap::DashMap;
use glob::glob;
use mj_io::{
    build_pbar, create_writer, expand_dirs, get_output_filename, read_pathbuf, read_pathbuf_to_mem,
    write_mem_to_pathbuf,
};
use ndarray::Array1;
use rand::Rng;
use rand::RngCore;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::Value as JSONValue;
use serde_yaml::Value;
use sha2::Digest;
use sha2::Sha256;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::fs;
use std::fs::create_dir_all;
use std::fs::OpenOptions;
use std::hash::BuildHasher;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::BufRead;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::panic::catch_unwind;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use tiktoken_rs::{cl100k_base, p50k_base, CoreBPE};
use unicode_segmentation::UnicodeSegmentation;

/*======================================================================
=                            FILE MAP STUFF                            =
======================================================================*/

/// Maps file paths to integer indices for efficient storage and reference.
///
/// In a single-node setting, the FileMap can be generated automatically from local files.
/// In a multi-node setting with remote storage (e.g., S3), use the Python tool
/// `python/file_map_builder.py` to generate the mapping.
///
/// ## Directory Structure Requirements
///
/// ### Single-Node (Local)
/// ```text
/// /local/path/data/
/// ├── file1.jsonl
/// ├── subdir/file2.jsonl.zst
/// └── file3.jsonl.gz
/// ```
///
/// ### Multi-Node (Remote + Local)
/// Remote storage should mirror local structure:
/// ```text
/// s3://bucket/prefix/data/     ->     /lfs/path/data/
/// ├── alpha.jsonl                      ├── alpha.jsonl
/// ├── beta/...                         ├── beta/...
/// └── sigma.jsonl.zst                  └── sigma.jsonl.zst
/// ```

#[derive(Serialize, Deserialize)]
pub struct FileMap {
    /// Local directory containing input files
    pub local_input: Option<PathBuf>,
    /// Remote directory path (e.g., S3 prefix)
    pub remote_input: Option<PathBuf>,
    /// Mapping from relative file path to integer index
    pub indices: HashMap<PathBuf, usize>,
}

impl FileMap {
    /// Creates a new FileMap by scanning the local input directory.
    ///
    /// # Arguments
    /// * `local_input` - Root directory containing all input files
    /// * `remote_input` - Optional remote storage path for reference
    ///
    /// # Returns
    /// A FileMap with all discovered `.jsonl`, `.jsonl.zst`, `.jsonl.zstd`, and `.jsonl.gz` files
    pub fn new(local_input: &PathBuf, remote_input: &Option<PathBuf>) -> Result<Self, Error> {
        // Implementation remains the same
        let input_vec = vec![local_input.clone()];
        let paths = expand_dirs(input_vec.clone(), None).unwrap();

        let stripped_paths: Vec<PathBuf> = paths
            .iter()
            .map(|p| p.strip_prefix(local_input.clone()).unwrap().to_path_buf())
            .collect();

        let indices: HashMap<PathBuf, usize> = stripped_paths
            .iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), i))
            .collect();
        Ok(FileMap {
            local_input: Some(local_input.clone()),
            remote_input: remote_input.clone(),
            indices,
        })
    }

    /// Serializes and saves the FileMap to disk as JSON.
    pub fn save(&self, save_loc: &PathBuf) -> Result<(), Error> {
        let json_bytes = serde_json::to_vec(self).unwrap();
        write_mem_to_pathbuf(&json_bytes, &save_loc)
    }

    /// Loads a FileMap from a saved JSON file.
    pub fn load(load_loc: &PathBuf) -> Result<Self, Error> {
        let json_bytes = read_pathbuf_to_mem(&load_loc).unwrap();
        let cursor = json_bytes.into_inner();
        let binding = cursor.into_inner();
        let contents = binding.as_slice();
        let filemap: FileMap = serde_json::from_slice(&contents).unwrap();
        Ok(filemap)
    }

    /// Returns a subset of paths assigned to a specific chunk for parallel processing.
    ///
    /// # Arguments
    /// * `chunk_id` - The chunk to retrieve (0 to num_chunks-1)
    /// * `num_chunks` - Total number of chunks for workload distribution
    ///
    /// # Returns
    /// Vector of (PathBuf, path_id) pairs for this chunk
    pub fn get_path_chunk(&self, chunk_id: usize, num_chunks: usize) -> Vec<(PathBuf, usize)> {
        let chunk: Vec<(PathBuf, usize)> = self
            .indices
            .iter()
            .filter(|(_k, v)| *v % num_chunks == chunk_id)
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        chunk
    }

    /// Returns the total number of files in the mapping.
    pub fn len(&self) -> usize {
        self.indices.len()
    }
}

/*======================================================================
=                            HASHING STUFF                             =
======================================================================*/

/// Unified tokenizer supporting multiple tokenization strategies.
///
/// Supported tokenizers:
/// - `"cl100k"` or `"cl100k_base"`: OpenAI's cl100k_base tokenizer
/// - `"p50k"`: OpenAI's p50k_base tokenizer  
/// - `"uniseg"`: Unicode word boundary segmentation
/// - `"bytes"`: Character-level (byte-based)
pub struct OmniTokenizer {
    tokenizer_name: String,
    inner: CoreBPE,
}

impl OmniTokenizer {
    /// Creates a new tokenizer with the specified strategy.    
    pub fn new(tokenizer_name: &str) -> Result<Self, Error> {
        // The default configuration and documented YAML use the full tokenizer name.
        let tokenizer_name = match tokenizer_name {
            "cl100k_base" => "cl100k",
            name => name,
        };
        // Validate tokenizer name
        match tokenizer_name {
            "p50k" | "cl100k" | "uniseg" | "bytes" => {
                // Valid tokenizer, proceed
            }
            _ => {
                return Err(Error::msg(format!(
                    "Unknown tokenizer: '{}'. Supported tokenizers are: p50k, cl100k, uniseg, bytes",
                    tokenizer_name
                )));
            }
        }

        if tokenizer_name == "cl100k" {
            Ok(OmniTokenizer {
                tokenizer_name: tokenizer_name.to_string(),
                inner: cl100k_base().unwrap(),
            })
        } else {
            Ok(OmniTokenizer {
                tokenizer_name: tokenizer_name.to_string(),
                inner: p50k_base().unwrap(),
            })
        }
    }

    /// Encodes text into a sequence of token IDs.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        match self.tokenizer_name.as_str() {
            "p50k" => self.inner.encode_with_special_tokens(text),
            "cl100k" => self.inner.encode_with_special_tokens(text),
            "uniseg" => text
                .split_word_bounds()
                .map(|w| {
                    let mut hasher = DefaultHasher::new();
                    w.hash(&mut hasher);
                    hasher.finish() as usize
                })
                .collect(),
            "bytes" => text.bytes().map(|b| b as usize).collect(),
            _ => {
                panic!(
                    "Unknown tokenizer: '{}'. Supported tokenizers are p50k, cl100k, uniseg, bytes",
                    self.tokenizer_name
                );
            }
        }
    }
}

/// Computes MinHash signatures for all documents in the assigned path chunk.
///
/// ## Output Structure
/// Signatures are stored in a hierarchical directory structure:
/// ```text
/// storage_dir/signatures/
/// └── band_XXX/
///     └── sigchunk_YYY/
///         └── pathchunk_ZZZ.sig.bin
/// ```
///
/// Each `.sig.bin` file contains packed binary data with entries of:
/// `(signature, path_id, line_num)`
///
/// ## Parallelism
/// This function processes files in parallel within a single path chunk.
/// Multiple instances can run concurrently on different path chunks.
///
/// # Arguments
/// * `config_obj` - MinHash configuration parameters
/// * `file_map` - Mapping of file paths to indices
/// * `storage_dir` - Root directory for output files
/// * `content_key` - JSON key containing document text
/// * `path_chunk` - This worker's chunk ID
/// * `num_path_chunks` - Total number of path chunks
/// * `id_subext` - Optional identifier suffix for output files
/// * `local_input` - Override for local input directory
pub fn hash_only(
    config_obj: &Config,
    file_map: &FileMap,
    storage_dir: &PathBuf,
    content_key: &String,
    path_chunk: usize,
    num_path_chunks: usize,
    id_subext: Option<String>,
    local_input: Option<PathBuf>,
) -> Result<(), Error> {
    println!(
        "Starting part of Minhash run | config {:?} | chunk {:?}/{:?}",
        config_obj, path_chunk, num_path_chunks
    );
    let start_main = Instant::now();

    // Initialize everything we need to hash...

    // -- Set up hashing stuff
    let band_seeds: Vec<u64> = _expand_rng(
        config_obj
            .minhash_params
            .permutation_seed
            .try_into()
            .unwrap(),
        config_obj.minhash_params.num_buckets,
    );
    // -- Get files to hash

    let local_input = if let Some(local_input) = local_input {
        local_input
    } else {
        file_map.local_input.clone().unwrap()
    };
    let this_chunk = file_map.get_path_chunk(path_chunk, num_path_chunks);
    let this_chunk: Vec<(PathBuf, usize)> = this_chunk
        .into_par_iter()
        .filter(|(path, _path_id)| local_input.join(path).exists())
        .collect();

    // -- Handle storage stuff
    let sig_storage = storage_dir.clone().join("sig_storage");
    create_dir_all(&sig_storage).unwrap();
    let num_sig_chunks = config_obj.eng_params.num_sig_chunks;
    let signature_writer = SignatureWriter::new(
        &sig_storage,
        band_seeds.clone(),
        config_obj.eng_params.num_sig_chunks,
        path_chunk,
        &id_subext,
    );
    let path_size = to_byte_size(file_map.indices.len());
    let line_size = to_byte_size(config_obj.eng_params.max_lines_per_path);
    let sig_size = compute_sig_size(config_obj.eng_params.num_docs);

    // And then loop through files and hash everything
    let start_hashing = Instant::now();
    let total_docs_hashed = AtomicUsize::new(0);
    let hash_pbar = build_pbar(this_chunk.len(), "Paths");

    this_chunk.par_iter().for_each(|(path, path_id)| {
        let docs_hashed = process_path(
            &local_input.join(path),
            &band_seeds,
            *path_id,
            config_obj.minhash_params.bucket_size,
            config_obj.minhash_params.ngram_size,
            config_obj.minhash_params.tokenizer.as_str(),
            &signature_writer,
            num_sig_chunks,
            path_size,
            line_size,
            sig_size,
            content_key,
        )
        .unwrap();
        total_docs_hashed.fetch_add(docs_hashed, Ordering::SeqCst);
        hash_pbar.inc(1);
    });
    signature_writer.finish().unwrap();
    println!(
        "(Chunk {:?}) ...collected all hashes in {:?} seconds",
        path_chunk,
        start_hashing.elapsed().as_secs()
    );
    println!("-------------------------");
    println!(
        "Completing part of Minhash run | config {:?} | chunk {:?}/{:?}",
        config_obj, path_chunk, num_path_chunks
    );
    println!("Computed hashes {:?} docs", total_docs_hashed.into_inner());
    println!("Total runtime: {:?} (s)", start_main.elapsed().as_secs());
    Ok(())
}

/// Processes a single file, computing MinHash signatures for all documents.
///
/// # Arguments
/// * `path` - Path to the input JSONL file
/// * `band_seeds` - Random seeds for each LSH band
/// * `path_id` - Integer identifier for this file
/// * `band_size` - Number of hash values per band
/// * `ngram_size` - Size of n-grams for shingling
/// * `tokenizer_str` - Name of tokenization strategy
/// * `signature_writer` - Writer for outputting signatures
/// * `num_sig_chunks` - Number of signature chunks for distribution
/// * `path_size` - Byte size for encoding path IDs
/// * `line_size` - Byte size for encoding line numbers
/// * `sig_size` - Byte size for encoding signatures
/// * `content_key` - JSON key containing document text
///
/// # Returns
/// Number of documents successfully hashed
fn process_path(
    path: &PathBuf,
    band_seeds: &Vec<u64>,
    path_id: usize,
    band_size: usize,
    ngram_size: usize,
    tokenizer_str: &str,
    signature_writer: &SignatureWriter,
    num_sig_chunks: usize,
    path_size: usize,
    line_size: usize,
    sig_size: usize,
    content_key: &str,
) -> Result<usize, Error> {
    // Setup things: load data, build tokenizer, etc
    let data = read_pathbuf(path, true).unwrap();
    // let mut buffer = Vec::new();
    // data.read_to_end(&mut buffer).unwrap();
    // println!("READ DATA {:?}", buffer);
    let tokenizer = OmniTokenizer::new(tokenizer_str).unwrap();
    let num_bands = band_seeds.len();
    let perm_seeds: Vec<u64> = band_seeds
        .iter()
        .flat_map(|seed| _expand_rng(*seed, band_size))
        .collect();
    let path_id = IntValueEnum::new(path_id, path_size);

    let mut docs_hashed = 0;
    for (line_num, line) in data.lines().enumerate() {
        let line = line.unwrap();
        let json_obj: Value = serde_json::from_str(&line).expect(&format!(
            "Failed to parse {:?} {:?}",
            path.clone(),
            line_num
        ));
        let line_num = IntValueEnum::new(line_num, line_size);
        let text = json_obj
            .get(content_key)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();

        let Ok(tokens) = catch_unwind(|| preprocess_text(&text, &tokenizer)) else {
            println!(
                "Tokenization failed on {:?} | {:?} | {:?}",
                path.clone(),
                path_id,
                line_num.as_uint::<usize>()
            );
            continue;
        };
        let hash_vals = get_hash_vals_from_tokens(tokens, &perm_seeds, ngram_size);
        docs_hashed += 1;

        let bands = hash_vals.into_shape((num_bands, band_size)).unwrap();
        for (row, band_seed) in bands.rows().into_iter().zip(band_seeds.iter()) {
            let mut hasher = Sha256::new();
            hasher.update(bytemuck::cast_slice(row.as_slice().unwrap()));
            let hash = hasher.finalize();
            let band_signature = IntValueEnum::from_bytes(hash[..sig_size].to_vec(), sig_size);
            _save_band_signature_to_disk(
                &signature_writer,
                *band_seed,
                band_signature,
                path_id.clone(),
                line_num.clone(),
                num_sig_chunks,
            )
            .unwrap();
        }
    }
    Ok(docs_hashed)
}

/// Preprocesses text by cleaning and tokenizing.
pub fn preprocess_text(text: &str, tokenizer: &OmniTokenizer) -> Vec<usize> {
    let text = clean_text(text);
    let tokens = tokenizer.encode(&text);
    tokens
}

fn clean_text(text: &str) -> String {
    // SlimPajama text cleaning process

    // Convert the document to lowercase
    let mut text = text.to_lowercase();

    // Remove punctuation
    let punctuation: &[_] = &[
        '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<',
        '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~',
    ];
    text.retain(|c| !punctuation.contains(&c));

    // Replace multiple whitespace characters with a single space
    let re = Regex::new(r"\s+").unwrap();
    text = re.replace_all(&text, " ").to_string();

    // Trim leading and trailing whitespace
    text.trim().to_string()
}

/// Computes MinHash values from a token sequence using n-gram shingling.
///
/// # Arguments
/// * `tokens` - Sequence of token IDs
/// * `perm_seeds` - Random seeds for hash permutations
/// * `ngram_size` - Size of n-grams (shingles)
///
/// # Returns
/// Array of minimum hash values across all permutations
pub fn get_hash_vals_from_tokens(
    tokens: Vec<usize>,
    perm_seeds: &Vec<u64>,
    ngram_size: usize,
) -> Array1<u64> {
    let a = _init_permutations(perm_seeds);
    let n = perm_seeds.len();
    let mut hash_vals = Array1::ones(n) * u64::MAX;
    let mut ngram: VecDeque<usize> = VecDeque::with_capacity(ngram_size);
    let mut ngram_count = 0;
    for token in tokens {
        ngram.push_back(token);
        if ngram.len() >= ngram_size {
            ngram_count += 1;
            hash_vals = _update_hash_vals(hash_vals, &a, &ngram);
            ngram.pop_front();
        }
    }
    hash_vals = if ngram_count == 0 {
        _update_hash_vals(hash_vals, &a, &ngram) // short document, still wanna hash it
    } else {
        hash_vals
    };

    hash_vals
}

/// Initializes random permutation coefficients for MinHash.
///
/// Each seed produces an odd coefficient for the hash function:
/// `h(x) = (a * x) mod 2^128`, where only the top 64 bits are used.
fn _init_permutations(seeds: &Vec<u64>) -> Array1<u128> {
    // Initialize the permutations needed for each minhash
    let n = seeds.len();
    let mut a = Array1::zeros(n);
    for (i, &seed) in seeds.iter().enumerate() {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut coeff = rng.gen::<u128>() as u128;
        if coeff % 2 == 0 {
            coeff += 1;
        }
        a[i] = coeff;
    }
    a
}

/// Updates minimum hash values with a new n-gram.
///
/// Uses two independent hash functions to create a 128-bit hash,
/// then multiplies by permutation coefficients.
fn _update_hash_vals(
    mut hash_vals: Array1<u64>,
    a: &Array1<u128>,
    ngram: &VecDeque<usize>,
) -> Array1<u64> {
    // hash the vecdeque as a u128
    let builder_a = RandomState::with_seeds(123, 456, 789, 101112);
    let mut hasher_a = builder_a.build_hasher();
    ngram.hash(&mut hasher_a);
    let hash_val_a = hasher_a.finish();

    let builder_b = RandomState::with_seeds(131415, 161718, 192021, 222324);
    let mut hasher_b = builder_b.build_hasher();
    ngram.hash(&mut hasher_b);
    let hash_val_b = hasher_b.finish();
    let cur_hash = ((hash_val_a as u128) << 64) | (hash_val_b as u128);

    // then multiply by a (mod 2^128) and take top 64 most significant bits
    let phv: Array1<u64> = a.mapv(|x| (x.wrapping_mul(cur_hash) >> 64) as u64);
    hash_vals.zip_mut_with(&phv, |x, y| *x = std::cmp::min(*x, *y));

    hash_vals
}

/// Expands a single seed into multiple random values using a PRNG.
fn _expand_rng(seed: u64, output_size: usize) -> Vec<u64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
    let mut output: Vec<u64> = Vec::new();
    for _i in 0..output_size {
        output.push(rng.next_u64());
    }
    output
}

/// Writes a band signature to disk in the appropriate signature chunk file.
fn _save_band_signature_to_disk(
    signature_writer: &SignatureWriter,
    band_seed: u64,
    band_signature: IntValueEnum,
    path_id: IntValueEnum,
    line_num: IntValueEnum,
    num_sig_chunks: usize,
) -> Result<(), Error> {
    let sig_chunk = band_signature.as_uint::<usize>() % num_sig_chunks;
    let contents = [
        band_signature.as_bytes(),
        path_id.as_bytes(),
        line_num.as_bytes(),
    ]
    .concat();
    signature_writer
        .write_line(band_seed, sig_chunk, contents)
        .unwrap();
    Ok(())
}

/*================================================================================
=                            EDGE GATHERING STUFF                                =
================================================================================*/
/* This creates the edge files.

    ----- edge files -----
    Edge file naming structure:
    working_dir/
    └── edges/
       └── sigchunk_YYY/
           └── band_XXX.edges.bin

    Where the:
        - band_id (XXX) ranges from 0..num_bands (pulled from signatures files)
        - sigchunk_id (YYY) ranges from 0..num_sigchunks (pulled from signatures files)

    And the contents of each file is a packed-bytes object, where if
    (path_A, line_1) and (path_B, line_2) have the same signature in a single band, then the bytes

        `(path_A, line_1, path_B, line_2)`

    appear in the file.


PARALLELISM STRATEGY: This is parallel across the band_ids of the signature files.
I.e., if you have many signature files that look like:
storage_dir/signatures/band_XXX/sigchunk_YYY/pathchunk_ZZZ.sig.bin

You should first gather all files with a given band_XXX onto a single machine and then run this function

*/

/// Gathers edges by grouping documents with matching signatures.
///
/// ## Input
/// Reads signature files from: `storage_dir/sig_storage/band_XXX/sigchunk_YYY/pathchunk_ZZZ.sig.bin`
///
/// ## Output
/// Creates edge files: `storage_dir/edges/sigchunk_YYY/band_XXX.edges.bin`
///
/// Each edge file contains sequences of (path_id, line_num) pairs that share
/// the same signature in a band, terminated by sentinel values (max_path, max_line).
///
/// ## Parallelism
/// Processes bands in parallel. All signature files for a given band must be
/// available on the same machine before calling this function.
pub fn gather_edges(
    config_obj: &Config,
    file_map: &FileMap,
    storage_dir: &PathBuf,
) -> Result<(), Error> {
    println!("Starting edge gather");
    let eng_config_obj = &config_obj.eng_params;
    let start_main = Instant::now();

    // Load the config and initialize things
    let path_size = to_byte_size(file_map.indices.len());
    let line_size = to_byte_size(eng_config_obj.max_lines_per_path);
    let sig_size = compute_sig_size(eng_config_obj.num_docs);

    // Gather the files into the proper groups (which should live in the same hash-space-universe)
    let edge_groups = gather_groups(storage_dir.clone().join("sig_storage")).unwrap(); //(sigchunk, band_id) -> [sigfile_a, sigfile_b, ...]
                                                                                       // Then build the cliques for each group of (sigchunk, band_id) -- across all files!

    println!("Starting edge collection...");
    let pbar = build_pbar(edge_groups.len(), "Band groups");
    edge_groups.par_iter().for_each(|entry| {
        let (sigchunk, band_id) = entry.key();
        let band_group = build_band_group(entry.value(), sig_size, path_size, line_size).unwrap();
        let output_filename = storage_dir
            .clone()
            .join("edges")
            .join(format!("sigchunk_{:08}", sigchunk))
            .join(format!("band_{:08}.edges.bin", band_id));
        save_band_group(band_group, output_filename, path_size, line_size).unwrap();
        pbar.inc(1);
    });

    println!(
        "... Gathered edges in {:?} seconds",
        start_main.elapsed().as_secs()
    );
    // And save these for future use

    Ok(())
}

/// Groups signature files by (sig_chunk, band_id) for parallel processing.
fn gather_groups(sig_storage: PathBuf) -> Result<DashMap<(usize, usize), Vec<PathBuf>>, Error> {
    let binding = sig_storage.clone().join("**").join("*.sig.bin");
    let sig_storage_str = binding.to_str().unwrap();
    let map: DashMap<(usize, usize), Vec<PathBuf>> = DashMap::new();
    for entry in glob(&sig_storage_str).unwrap() {
        let entry = entry.unwrap();
        let sigchunk_dir = entry
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|name| name.to_str())
            .unwrap();
        let sigchunk = sigchunk_dir
            .split('_')
            .last()
            .unwrap()
            .parse::<usize>()
            .unwrap();

        let band_id_dir = entry
            .parent()
            .unwrap()
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|name| name.to_str())
            .unwrap();
        let band_id = band_id_dir
            .split('_')
            .last()
            .unwrap()
            .parse::<usize>()
            .unwrap();

        map.entry((sigchunk, band_id)).or_default().push(entry);
    }

    Ok(map)
}

/// Builds collision groups for a single band from multiple signature files.
///
/// Returns a vector of cliques, where each clique is a list of (path_id, line_num)
/// pairs that have identical signatures in this band.
fn build_band_group(
    band_sigs: &Vec<PathBuf>,
    sig_size: usize,
    path_size: usize,
    line_size: usize,
) -> Result<Vec<Vec<(usize, usize)>>, Error> {
    // For a group of files that contain signatures within the same band (and a sig chunk)
    // Collects a list of (path_id: usize, line_id: usize) for each clique
    let entry_size = sig_size + path_size + line_size;

    // build map from signature -> [(path_id, line_num), ...]
    // to collect docs that have the same signature within this band
    let group_map: DashMap<IntValueEnum, Vec<(usize, usize)>> = DashMap::new();

    band_sigs.iter().for_each(|path| {
        let contents = read_pathbuf_to_mem(path).unwrap().into_inner().into_inner();
        contents.chunks(entry_size).for_each(|entry| {
            let sig = IntValueEnum::from_bytes(entry[..sig_size].to_vec(), sig_size);
            let path_id =
                IntValueEnum::from_bytes(entry[sig_size..sig_size + path_size].to_vec(), path_size)
                    .as_uint::<usize>();
            let line_id =
                IntValueEnum::from_bytes(entry[sig_size + path_size..].to_vec(), line_size)
                    .as_uint::<usize>();
            group_map.entry(sig).or_default().push((path_id, line_id));
        });
    });

    // Select only the groups that have size > 1
    let band_group: Vec<Vec<(usize, usize)>> = group_map
        .into_iter()
        .map(|(_, group)| group)
        .filter(|value| value.len() > 1)
        .collect();

    Ok(band_group)
}

/// Saves a band's collision groups to an edge file.
///
/// Each group is written as a sequence of (path_id, line_num) pairs,
/// terminated by (max_path, max_line) sentinels.
fn save_band_group(
    band_group: Vec<Vec<(usize, usize)>>,
    output_file: PathBuf,
    path_size: usize,
    line_size: usize,
) -> Result<(), Error> {
    let max_path = IntValueEnum::new(1 << path_size - 1, path_size);
    let max_line = IntValueEnum::new(1 << line_size - 1, line_size);
    let group_end = [max_path.as_bytes(), max_line.as_bytes()].concat();

    if let Some(parent_dir) = output_file.parent() {
        if !parent_dir.exists() {
            create_dir_all(parent_dir).unwrap();
        }
    }
    let mut writer = OpenOptions::new()
        .append(true)
        .create(true)
        .mode(0o644)
        .open(output_file)
        .unwrap();

    band_group.into_iter().for_each(|edgelist| {
        edgelist.into_iter().for_each(|(path_id, line_num)| {
            let contents = [
                IntValueEnum::new(path_id, path_size).as_bytes(),
                IntValueEnum::new(line_num, line_size).as_bytes(),
            ]
            .concat();
            writer.write_all(&contents).unwrap();
        });
        writer.write(&group_end.clone()).unwrap();
    });
    writer.flush().unwrap();
    Ok(())
}

/*======================================================================
=                            UNION FIND STUFF                          =
======================================================================*/
/* NO MULTINODE PARALLELISM! =(
This creates a union find structure to merge together the edges.
It creates the "cleaning" files used to annotate and/or deduplicate the data.
This also sets things up to be run on a path-wise multinode parallelism in the next step:

The files live in a structure like:
----- clean files -----
working_dir/
└── clean/
   └── chunk_ZZZ.clean.bin


The file structure of these looks like:
preamble: (40 bytes) like:
    [path_size, line_size, cc_id_size, cc_size_byte_size, cc_size_byte_size], each being u64s
And then repeated sequences of size (path_size + line_size + cc_id_size + 2*cc_size_byte_size) where each entry
is of the form: (path, line_num, cc_id, cc_size, cc_idx)

This lets us either:
i) annotate each doc ((path, line_num)) with a (cc_id, cc_size, cc_idx)
    (and then from here we can do more sophisticated things, like group/sort/etc)
ii) remove any doc with a cc_idx > 1 (effectively deduplicating)


PARALLELISM STRATEGY: UNABLE -- NEEDS TO BE RUN GLOBALLY!
*/

/// Builds a Union-Find structure to identify connected components of duplicates.
///
/// ## Output Structure
/// Creates cleaning metadata files: `storage_dir/clean/chunk_ZZZ.clean.bin`
///
/// Each file has a 40-byte header (5 × u64 little-endian):
/// ```text
/// [path_size, line_size, cc_id_size, cc_size_byte_size, cc_size_byte_size]
/// ```
///
/// Followed by rows of: `(path_id, line_num, cc_id, cc_size, cc_idx)`
/// where:
/// - `cc_id`: Connected component identifier
/// - `cc_size`: Total number of documents in this component
/// - `cc_idx`: Index within component (0 = first occurrence)
///
/// ## Parallelism
/// **WARNING**: This function must run on a single machine with access to ALL edge files.
/// No multi-node parallelism is supported for this stage.
pub fn build_uf(
    config_obj: &Config,
    file_map: &FileMap,
    storage_dir: &PathBuf,
    num_path_chunks: usize,
) -> Result<(), Error> {
    // Takes the edges (saved as lists of lists of (path_id, line_num) pairs)
    // and builds a union find object and then collects CC's and saves a list of ccs.
    // Unless otherwise specified, also creates a list of to-delete lines, grouped by path_id

    println!("Building UnionFind...");
    let start_main = Instant::now();
    let eng_config_obj = &config_obj.eng_params;

    // Load the config to initialize things
    let path_size = to_byte_size(file_map.indices.len());
    let line_size = to_byte_size(eng_config_obj.max_lines_per_path);

    // Build the union find and unite all the edges
    let uf = UFRush::new();
    let all_edge_files = expand_dirs(
        vec![storage_dir.clone().join("edges")],
        Some(vec![".edges.bin"].as_slice()),
    )
    .unwrap();
    println!("Adding edges to UF...");
    let pbar = build_pbar(all_edge_files.len(), "Edge files");
    all_edge_files.into_par_iter().for_each(|p| {
        add_edge_file_to_uf(&p, &uf, path_size, line_size).unwrap();
        pbar.inc(1);
    });

    // And then compress all paths in the union find
    println!("Compressing paths...");
    let keys: Vec<usize> = uf.nodes.par_iter().map(|entry| *entry.key()).collect();
    let pbar = build_pbar(keys.len(), "Compressing UF Paths...");
    keys.into_par_iter().for_each(|k| {
        uf.find_path_compression(k);
        pbar.inc(1);
    });
    println!(
        "Built unionfind in {:?} secs",
        start_main.elapsed().as_secs()
    );

    // And then get size of each cc
    let start_cc_size = Instant::now();
    println!("Computing CC Sizes");
    let cc_sizes = get_cc_sizes(&uf.nodes);
    println!(
        "Made CC Sizes in {:?} secs",
        start_cc_size.elapsed().as_secs()
    );

    // And then do the annotation/pruning stuff
    let start_prune_metadata = Instant::now();
    println!("Starting generation of pruning metadata...");
    make_pruning_metadata(
        uf.nodes,
        cc_sizes,
        &storage_dir,
        &file_map,
        num_path_chunks,
        path_size,
        line_size,
    )
    .unwrap();
    println!(
        "Made pruning metadata in {:?} secs",
        start_prune_metadata.elapsed().as_secs()
    );

    return Ok(());
}

/// Adds all edges from a single edge file to the Union-Find structure.
fn add_edge_file_to_uf(
    edge_file: &PathBuf,
    uf: &UFRush,
    path_size: usize,
    line_size: usize,
) -> Result<(), Error> {
    let edge_data = read_pathbuf_to_mem(edge_file)
        .unwrap()
        .into_inner()
        .into_inner();
    let max_path = IntValueEnum::new(1 << path_size - 1, path_size).as_uint::<usize>();
    let max_line = IntValueEnum::new(1 << line_size - 1, line_size).as_uint::<usize>();
    let group_end_id = pair2docid((max_path, max_line), line_size);
    let mut last_id = group_end_id;

    edge_data.chunks_exact(path_size + line_size).for_each(|c| {
        let path_id =
            IntValueEnum::from_bytes(c[..path_size].to_vec(), path_size).as_uint::<usize>();
        let line_num =
            IntValueEnum::from_bytes(c[path_size..].to_vec(), line_size).as_uint::<usize>();
        let cur_id = pair2docid((path_id, line_num), line_size);
        if cur_id != group_end_id && last_id != group_end_id {
            uf.unite(last_id, cur_id);
        }
        last_id = cur_id;
    });

    Ok(())
}

/// Converts a (path_id, line_id) pair to a single integer for Union-Find.
fn pair2docid(pair: (usize, usize), line_size: usize) -> usize {
    // Given a (path_id, line_id) pair, converts it into a single usize
    // (which is needed for UF rush)
    let (path_id, line_id) = pair;
    (path_id << (line_size * 8)) + line_id
}

/// Converts a Union-Find document ID back to (path_id, line_id).
fn docid2pair(docid: usize, line_size: usize) -> (usize, usize) {
    // Inverse function of the pair2docid
    let mask = (1 << (line_size * 8)) - 1;
    (docid >> (line_size * 8), docid & mask)
}

/// Computes the size of each connected component in the Union-Find structure.
fn get_cc_sizes(uf_nodes: &DashMap<usize, AtomicUsize>) -> DashMap<usize, usize> {
    let cc_sizes: DashMap<usize, usize> = DashMap::new();
    let pbar = build_pbar(uf_nodes.len(), "Building cc sizes");
    uf_nodes.par_iter().for_each(|entry| {
        let val = entry.value().load(Ordering::Relaxed);
        let cc_id = uf_parent(val);
        let _ = *cc_sizes
            .entry(cc_id)
            .and_modify(|count| *count += 1)
            .or_insert(1);
        pbar.inc(1);
    });
    cc_sizes
}

/// Generates pruning metadata files for each path chunk.
///
/// For each document in the Union-Find, writes:
/// - Which connected component it belongs to
/// - The size of that component
/// - Its index within the component (for determining which copy to keep)
fn make_pruning_metadata(
    uf_nodes: DashMap<usize, AtomicUsize>,
    cc_sizes: DashMap<usize, usize>,
    working_dir: &PathBuf,
    file_map: &FileMap,
    num_path_chunks: usize,
    path_size: usize,
    line_size: usize,
) -> Result<(), Error> {
    /*
    Makes a bunch of pruning metadata files:
    each file has a 40byte preamble with 5 x 8byte-le headers of [from IntValueEnum!]
    [path_size, line_size, cc_id_size, cc_size_byte_size, cc_size_byte_size]
    and then just a list of bytes of this^ form

    NOTE: - headers are little-endian
          - rows are from IntValueEnum (use this API!)

    */

    // Make output directories and writers for these
    let clean_dir = working_dir.clone().join("clean");
    let max_cc_size = cc_sizes.par_iter().map(|e| *e.value()).max().unwrap_or(1);
    let max_cc_id = cc_sizes.par_iter().map(|e| *e.key()).max().unwrap_or(1);
    let cc_size_byte_size = to_byte_size(max_cc_size);
    let cc_id_byte_size = to_byte_size(max_cc_id);

    // Map path id to chunk id
    let path_id_2_chunk_id: DashMap<usize, usize> = DashMap::new();
    for chunk_id in 0..num_path_chunks {
        let path_chunk = file_map.get_path_chunk(chunk_id, num_path_chunks);
        path_chunk.par_iter().for_each(|entry| {
            let path_id = entry.1 as usize;
            path_id_2_chunk_id.insert(path_id, chunk_id);
        });
    }

    let cc_idxs: DashMap<usize, usize> = DashMap::new();
    let clean_writer = GenWriter::new(
        &clean_dir,
        num_path_chunks,
        "clean",
        &None,
        false,
        Some(usize::MAX),
    );
    let metadata_header: Vec<u8> = vec![
        (path_size as u64).to_le_bytes(),
        (line_size as u64).to_le_bytes(),
        (cc_id_byte_size as u64).to_le_bytes(),
        (cc_size_byte_size as u64).to_le_bytes(),
        (cc_size_byte_size as u64).to_le_bytes(),
    ]
    .into_iter()
    .flat_map(|s| s)
    .collect();
    (0..num_path_chunks).into_par_iter().for_each(|i| {
        clean_writer
            .write_line(0, metadata_header.clone(), Some(i))
            .unwrap();
    });

    // And then loop through nodes
    let pbar = build_pbar(uf_nodes.len(), "Writing clean metadata");
    uf_nodes.into_par_iter().for_each(|(child, v)| {
        let cc_id = uf_parent(v.into_inner());
        let cc_size = *cc_sizes.get(&cc_id).unwrap();
        let cc_idx = *cc_idxs
            .entry(cc_id)
            .and_modify(|count| *count += 1)
            .or_insert(0);
        let (child_path, child_line) = docid2pair(child, line_size);
        let path_chunk = *path_id_2_chunk_id.get(&child_path).unwrap().value();
        let child_path_bytes = IntValueEnum::new(child_path, path_size).as_bytes().to_vec();
        let child_line_bytes = IntValueEnum::new(child_line, line_size).as_bytes().to_vec();
        let cc_id_bytes = IntValueEnum::new(cc_id, cc_id_byte_size)
            .as_bytes()
            .to_vec();
        let cc_size_bytes = IntValueEnum::new(cc_size, cc_size_byte_size)
            .as_bytes()
            .to_vec();
        let cc_idx_bytes = IntValueEnum::new(cc_idx, cc_size_byte_size)
            .as_bytes()
            .to_vec();

        // metadata file has lines of (child_path, child_line, cc_id, cc_size, cc_idx)
        let contents = vec![
            child_path_bytes,
            child_line_bytes,
            cc_id_bytes,
            cc_size_bytes,
            cc_idx_bytes,
        ]
        .into_iter()
        .flat_map(|s| s)
        .collect();
        clean_writer
            .write_line(0, contents, Some(path_chunk))
            .unwrap();
        pbar.inc(1);
    });
    clean_writer.finish().unwrap();
    Ok(())
}

/*======================================================================
=                            CLEANING STUFF                            =
======================================================================*/
/*
Actually modifies the input data by either removing duplicates or annotating documents with duplicate information.
Requires the .clean.bin files from the union find step

Will either annotate or remove duplicates (or both)


PARALLELISM STRATEGY: This is parallel across slices of _paths_ of data
*/
/// Applies deduplication by annotating and/or removing duplicates from input files.
///
/// Uses the cleaning metadata from the Union-Find stage to:
/// 1. **Annotate**: Add duplicate information to each document as JSON
/// 2. **Remove**: Delete all but the first occurrence of each duplicate
///
/// ## Parallelism
/// Processes files in parallel within a single path chunk.
/// Multiple instances can run concurrently on different path chunks.
///
/// # Arguments
/// * `config_obj` - Configuration including output options
/// * `file_map` - File path mappings
/// * `input_dir` - Directory containing original files
/// * `storage_dir` - Directory containing cleaning metadata
/// * `output_dir` - Directory for deduplicated output
/// * `path_chunk` - This worker's chunk ID
/// * `num_path_chunks` - Total number of path chunks
pub fn clean_files(
    config_obj: &Config,
    file_map: &FileMap,
    input_dir: &PathBuf,
    storage_dir: &PathBuf,
    output_dir: &PathBuf,
    path_chunk: usize,
    num_path_chunks: usize,
) -> Result<(), Error> {
    println!("Starting UF-based pruning...");
    let start_main = Instant::now();
    let output_config_obj = &config_obj.output_params;

    let metadata_dir = storage_dir.clone().join("clean");
    let path_chunk_files = file_map.get_path_chunk(path_chunk, num_path_chunks);
    let path_chunk_files: Vec<(PathBuf, usize)> = path_chunk_files
        .into_par_iter()
        .filter(|(path, _path_id)| input_dir.join(path).exists())
        .collect();
    // Parse the metadata into a map from path_id -> [(line_num, cc_id, cc_size, cc_idx),...]
    println!("Reading metadata file from disk...");
    let start_clean_read = Instant::now();
    let metadata_file = GenWriter::get_filename(&metadata_dir, path_chunk, 0, "clean", &None);
    let metadata = parse_clean_metadata_file(&metadata_file).unwrap();
    println!(
        "Parsed metadata file in {:?} seconds",
        start_clean_read.elapsed().as_secs()
    );

    println!("Scrubbing files...");
    let start_clean = Instant::now();
    let documents_removed = AtomicUsize::new(0);
    let documents_seen = AtomicUsize::new(0);
    let pbar = build_pbar(path_chunk_files.len(), "Files to clean");

    path_chunk_files
        .into_par_iter()
        .for_each(|(path, path_id)| {
            let line_data = metadata.remove(&path_id).unwrap_or_default().1;
            let (lines_seen, lines_removed) = clean_path(
                &input_dir.clone().join(&path),
                line_data,
                &input_dir,
                &output_dir,
                output_config_obj.annotate,
                &output_config_obj.annotate_key,
                output_config_obj.remove_duplicates,
            )
            .unwrap();
            if output_config_obj.delete_while_cleaning {
                fs::remove_file(&input_dir.clone().join(path)).unwrap();
            }
            documents_removed.fetch_add(lines_removed, Ordering::Relaxed);
            documents_seen.fetch_add(lines_seen, Ordering::Relaxed);
            pbar.inc(1);
        });

    let documents_seen = documents_seen.into_inner();
    let documents_removed = documents_removed.into_inner();
    println!(
        "Scrubbed files in {:?} secs",
        start_clean.elapsed().as_secs()
    );
    println!(
        "Processed all files in {:?} secs",
        start_main.elapsed().as_secs()
    );
    println!("Saw {:?} docs", documents_seen);
    println!("Removed {:?} docs", documents_removed);
    println!(
        "Removal rate would be {:?}%",
        100.0 * documents_removed as f32 / (documents_seen as f32)
    );
    Ok(())
}

/// Processes a single file, applying annotation and/or deduplication.
///
/// # Returns
/// Tuple of (lines_seen, lines_removed) for statistics
fn clean_path(
    input_path: &PathBuf,
    line_data: Vec<(usize, usize, usize, usize)>,
    input_dir: &PathBuf,
    output_dir: &PathBuf,
    annotate: bool,
    annotate_key: &String,
    do_remove: bool,
) -> Result<(usize, usize), Error> {
    let output_filename = get_output_filename(input_path, input_dir, output_dir).unwrap();
    let contents = read_pathbuf_to_mem(input_path).unwrap();

    // Line_num -> (cc_id, cc_size, cc_idx)
    let anno_lookup: HashMap<usize, (usize, usize, usize)> = line_data
        .into_iter()
        .map(|(a, b, c, d)| (a, (b, c, d)))
        .collect();

    //let mut concat_kill: HashMap<Vec<String>, (usize, usize, usize)> = HashMap::new();
    let mut lines_seen = 0;
    let mut lines_removed = 0;

    fn get_anno_value(val: (usize, usize, usize)) -> JSONValue {
        json!({"cc_id": val.0,
               "cc_size": val.1,
               "cc_idx": val.2})
    }
    let mut writer = create_writer(&output_filename).unwrap();
    for (line_num, line) in contents.lines().enumerate() {
        lines_seen += 1;
        let line = line?;
        if anno_lookup.contains_key(&line_num) {
            // Need to either annotate and write or just remove
            let (cc_id, cc_size, cc_idx) = *anno_lookup.get(&line_num).unwrap();
            if cc_idx > 0 {
                lines_removed += 1;
            }
            // Remove if not the first idx
            if cc_idx > 0 && do_remove {
                continue;
            }

            if annotate {
                // If we want to annotate, go ahead and do that
                let mut line_json: JSONValue = serde_json::from_str(&line).unwrap();
                let anno_value = get_anno_value((cc_id, cc_size, cc_idx));
                json_set(&mut line_json, &annotate_key.clone(), anno_value).unwrap();
                writer
                    .write_line(&serde_json::to_vec(&line_json).unwrap())
                    .unwrap();
            } else {
                // Otherwise just write the line
                writer.write_line(&line.as_bytes().to_vec()).unwrap();
            }
        } else {
            // Not found in minhash, is "unique" and gets to survive
            writer.write_line(&line.as_bytes().to_vec()).unwrap();
        }
    }

    Ok((lines_seen, lines_removed))
}

/// Parses a cleaning metadata file into a map from path_id to duplicate information.
///
/// # Returns
/// Map of path_id -> Vec<(line_num, cc_id, cc_size, cc_idx)>
fn parse_clean_metadata_file(
    clean_file: &PathBuf,
) -> Result<DashMap<usize, Vec<(usize, usize, usize, usize)>>, Error> {
    let contents = read_pathbuf_to_mem(clean_file)
        .unwrap()
        .into_inner()
        .into_inner();
    /*
    Loads the metadata_header file
    Header of 5xu64-le
    And then rows are from IntValueEnums
    Rows are of form: (path_id, line_num, cc_id, cc_size, cc_idx)
    */
    const HEADER_SIZE: usize = 5 * 8;
    let path_size = u64::from_le_bytes(contents[0 * 8..0 * 8 + 8].try_into().unwrap()) as usize;
    let line_size = u64::from_le_bytes(contents[1 * 8..1 * 8 + 8].try_into().unwrap()) as usize;
    let cc_id_byte_size =
        u64::from_le_bytes(contents[2 * 8..2 * 8 + 8].try_into().unwrap()) as usize;
    let cc_size_byte_size =
        u64::from_le_bytes(contents[3 * 8..3 * 8 + 8].try_into().unwrap()) as usize;
    let entry_size = path_size + line_size + cc_id_byte_size + 2 * cc_size_byte_size;
    let chunk_count = (contents.len() - HEADER_SIZE) / entry_size;
    let pbar = build_pbar(chunk_count, "Reading metadata file");

    let metadata: DashMap<usize, Vec<(usize, usize, usize, usize)>> = DashMap::new();
    // path_id -> [(line_num, cc_id, cc_size, cc_idx)]

    (0..chunk_count).into_par_iter().for_each(|idx| {
        let start_idx = idx * entry_size + HEADER_SIZE;
        let chunk = contents[start_idx..start_idx + entry_size].to_vec();
        let path_bytes = chunk[..path_size].to_vec();
        let line_bytes = chunk[path_size..path_size + line_size].to_vec();
        let cc_id_bytes =
            chunk[path_size + line_size..path_size + line_size + cc_id_byte_size].to_vec();
        let cc_size_bytes = chunk[path_size + line_size + cc_id_byte_size
            ..path_size + line_size + cc_id_byte_size + cc_size_byte_size]
            .to_vec();
        let cc_idx_bytes =
            chunk[path_size + line_size + cc_id_byte_size + cc_size_byte_size..].to_vec();

        let path_id = IntValueEnum::from_bytes(path_bytes, path_size).as_uint::<usize>();
        let line_num = IntValueEnum::from_bytes(line_bytes, line_size).as_uint::<usize>();
        let cc_id = IntValueEnum::from_bytes(cc_id_bytes, cc_id_byte_size).as_uint::<usize>();
        let cc_size = IntValueEnum::from_bytes(cc_size_bytes, cc_size_byte_size).as_uint::<usize>();
        let cc_idx = IntValueEnum::from_bytes(cc_idx_bytes, cc_size_byte_size).as_uint::<usize>();

        metadata
            .entry(path_id)
            .or_default()
            .push((line_num, cc_id, cc_size, cc_idx));
        pbar.inc(1);
    });

    Ok(metadata)
}

#[cfg(test)]
mod tokenizer_tests {
    use super::OmniTokenizer;
    use crate::minhash_config::Config;

    #[test]
    fn default_tokenizer_matches_supported_short_name() {
        let config = Config::default();
        let default = OmniTokenizer::new(&config.minhash_params.tokenizer).unwrap();
        let explicit = OmniTokenizer::new("cl100k").unwrap();
        let text = "A short example with punctuation: 2 + 3 = 5.";
        assert_eq!(default.encode(text), explicit.encode(text));
    }

    #[test]
    fn unknown_tokenizer_is_rejected() {
        assert!(OmniTokenizer::new("not-a-tokenizer").is_err());
    }
}
