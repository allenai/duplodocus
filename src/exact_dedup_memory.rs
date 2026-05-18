//! # In-Memory Exact Deduplication
//!
//! Fast, parallel exact deduplication of JSONL files using xxHash3 for duplicate detection.
//!
//! ## Overview
//!
//! This module removes documents with identical content by computing hash values
//! and tracking which hashes have been seen. Uses xxHash3 for fast, high-quality
//! hashing with minimal collisions.
//!
//! ## Features
//!
//! - **Fast hashing**: xxHash3 is one of the fastest non-cryptographic hash functions
//! - **Parallel processing**: Multi-threaded file processing using Rayon
//! - **Flexible hash sizes**: 64-bit (good for ~4 billion documents) or 128-bit (virtually unlimited)
//! - **Pre-computed hashes**: Can use existing hash fields instead of recomputing
//! - **Annotation mode**: Add duplicate metadata instead of removing documents
//!
//! ## Hash Size Selection
//!
//! - **64-bit**: Sufficient for most use cases (up to ~4 billion documents with low collision risk)
//! - **128-bit**: Use for massive datasets or when collision risk must be minimized
//!
//! Birthday paradox collision probability:
//! - 64-bit: 50% collision chance at ~5 billion documents
//! - 128-bit: 50% collision chance at ~2^64 documents (not a practical concern)

use anyhow::{anyhow, Error, Result};
use dashmap::DashMap;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::fmt::Debug;
use std::hash::Hash;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use xxhash_rust::xxh3::{xxh3_128, xxh3_64};

use crate::utils::{json_get, json_set};
use mj_io::{build_pbar, create_writer, expand_dirs, get_output_filename, read_pathbuf};
use std::time::Instant;
/*
EXACT DEDUPLICATION MODULE

Fast, parallel deduplication of JSONL files using xxHash3 for duplicate detection.

## Usage
- Processes directories of JSON Lines files
- Removes duplicates based on text content or pre-computed hashes
- Supports 64-bit and 128-bit hash modes
- Uses parallel processing with thread-safe hash counting

## Parameters
- `text_key`: JSON field containing text content to deduplicate
- `hash_key`: Optional field with pre-computed hashes (skips hashing step)
- `hash_bits`: 64 or 128 bit hash size (64-bit sufficient for ~4B documents)
*/

/*=======================================================
=                     HELPER DATA TYPES                 =
=======================================================*/
/// Trait for document hash values supporting both 64-bit and 128-bit modes.
///
/// Provides a unified interface for hashing strings and converting between
/// JSON and native types.
trait DocHash: Copy + Eq + Hash + Debug + Send + Sync + 'static {
    fn hash_string(text: &String) -> Self;
    fn from_json(value: &Value) -> Result<Self, Error>;
    fn to_json(&self) -> Value;
}

/// 64-bit hash implementation using xxHash3.
///
/// Sufficient for datasets up to ~4 billion documents with low collision risk.
/// JSON representation: numeric value
impl DocHash for u64 {
    fn hash_string(text: &String) -> Self {
        xxh3_64(text.as_bytes())
    }
    fn from_json(value: &Value) -> Result<Self, Error> {
        Ok(value.as_u64().unwrap())
    }
    fn to_json(&self) -> Value {
        Value::Number((*self).into())
    }
}

/// 128-bit hash implementation using xxHash3.
///
/// Virtually eliminates collision risk for any practical dataset size.
/// JSON representation: string (since JSON doesn't support 128-bit integers)
impl DocHash for u128 {
    fn hash_string(text: &String) -> Self {
        xxh3_128(text.as_bytes())
    }
    fn from_json(value: &Value) -> Result<Self, Error> {
        Ok(value.as_str().unwrap().parse::<u128>().unwrap())
    }
    fn to_json(&self) -> Value {
        Value::String(self.to_string()) // u128 as string since JSON doesn't support 128-bit ints
    }
}

/*=====================================================
=                      MAIN FXN                       =
=====================================================*/

/// Performs exact deduplication on a directory of JSONL files.
///
/// Processes all files in parallel, keeping only the first occurrence of each
/// unique document. Documents are considered duplicates if they have identical
/// hash values.
///
/// # Arguments
///
/// * `input_dir` - Directory containing `.jsonl`, `.jsonl.zst`, `.jsonl.zstd`, or `.jsonl.gz` files
/// * `output_dir` - Directory where deduplicated files will be written
/// * `text_key` - JSON key containing document text to hash (e.g., "text", "content")
/// * `hash_key` - Optional JSON key with pre-computed hash values (skips hashing if provided)
/// * `hash_bits` - Hash size: 64 or 128 bits
/// * `annotate` - Optional JSON key for adding duplicate metadata instead of removing
///
/// # Hash Key Format
///
/// If `hash_key` is provided:
/// - For 64-bit: Must be a JSON number
/// - For 128-bit: Must be a JSON string containing a decimal integer
///
/// # Annotation Format
///
/// When `annotate` is Some, documents receive metadata instead of being removed:
/// ```json
/// {
///   "your_annotation_key": {
///     "hash": "12345678901234567890",  // The document's hash
///     "num_dups": 3                     // Total occurrences including this one
///   }
/// }
/// ```
///
/// # Returns
///
/// `Ok(())` on success, printing statistics:
/// - Total documents processed
/// - Documents kept
/// - Removal rate percentage
///
/// # Errors
///
/// Returns an error if:
/// - `hash_bits` is not 64 or 128
/// - Input directory doesn't exist or contains no valid files
/// - JSON parsing fails
/// - File I/O errors occur
///
/// # Performance
///
/// - **Speed**: Processes millions of documents per minute on modern hardware
/// - **Memory**: Requires ~24 bytes per unique document (hash + counter) in 64-bit mode,
///   ~40 bytes in 128-bit mode
/// - **Parallelism**: Automatically uses all available CPU cores
///
/// # Notes
///
/// - First occurrence of each document is always kept (or gets `num_dups` in annotation mode)
/// - Input files are never modified; all output goes to `output_dir`
/// - Output directory structure mirrors input directory structure
/// - Progress bar shows file processing status
pub fn exact_dedup_memory(
    input_dir: &PathBuf,
    output_dir: &PathBuf,
    text_key: &String,
    hash_key: Option<String>,
    hash_bits: usize,
    annotate: Option<String>,
    exclude_substring: String,
    include_substring: String,
) -> Result<(), Error> {
    let start_main = Instant::now();
    println!("Starting exact deduplication");

    let (seen_docs, kept_docs) = match hash_bits {
        64 => exact_dedup_impl::<u64>(input_dir, output_dir, text_key, hash_key, annotate, &exclude_substring, &include_substring).unwrap(),
        128 => {
            exact_dedup_impl::<u128>(input_dir, output_dir, text_key, hash_key, annotate, &exclude_substring, &include_substring).unwrap()
        }
        _ => {
            return Err(anyhow!(
                "Hash bits can only be 64 or 128, not {:?}",
                hash_bits
            ));
        }
    };
    let removal_rate = 100.0 * ((seen_docs - kept_docs) as f32) / seen_docs as f32;
    println!(
        "Finished exact deduplication in {:?} seconds",
        start_main.elapsed().as_secs()
    );
    println!(
        "Saw {:?} documents, and kept {:?} of them",
        seen_docs, kept_docs
    );
    println!("Removal rate was {:.2}%", removal_rate);

    Ok(())
}

/// Internal implementation of exact deduplication, generic over hash type.
///
/// This function is called by `exact_dedup_memory` with the appropriate hash type
/// based on the `hash_bits` parameter.
fn exact_dedup_impl<K: DocHash>(
    input_dir: &PathBuf,
    output_dir: &PathBuf,
    text_key: &String,
    hash_key: Option<String>,
    annotate: Option<String>,
    exclude_substring: &str,
    include_substring: &str,
) -> Result<(usize, usize), Error> {
    let raw_input_paths = expand_dirs(vec![input_dir.clone()], None).unwrap();
    let n_before = raw_input_paths.len();
    let input_paths: Vec<_> = raw_input_paths
        .into_iter()
        .filter(|p| {
            let s = p.to_string_lossy();
            (include_substring.is_empty() || s.contains(include_substring))
                && (exclude_substring.is_empty() || !s.contains(exclude_substring))
        })
        .collect();
    if !include_substring.is_empty() || !exclude_substring.is_empty() {
        println!(
            "Path filter: kept {} of {} (include={:?}, exclude={:?})",
            input_paths.len(),
            n_before,
            include_substring,
            exclude_substring
        );
    }
    let seen_docs = AtomicUsize::new(0);
    let kept_docs = AtomicUsize::new(0);
    let counter: DashMap<K, usize> = DashMap::new();
    if let Some(ref _annotate_key) = annotate {
        let anno_pbar = build_pbar(input_paths.len(), "Paths");
        input_paths.par_iter().for_each(|p| {
            build_out_counter(p, text_key, &hash_key, &counter).unwrap();
            anno_pbar.inc(1);
        })
    };

    let pbar = build_pbar(input_paths.len(), "Paths");
    let read_err_files = AtomicUsize::new(0);
    let json_err_lines = AtomicUsize::new(0);
    let hash_err_lines = AtomicUsize::new(0);
    input_paths.into_par_iter().for_each(|p| {
        let output_filename = get_output_filename(&p, input_dir, output_dir).unwrap();
        let (p_seen, p_kept) = exact_dedup_file(
            p, output_filename, text_key, &hash_key, &counter, &annotate,
            &read_err_files, &json_err_lines, &hash_err_lines,
        ).unwrap();
        seen_docs.fetch_add(p_seen, Ordering::Relaxed);
        kept_docs.fetch_add(p_kept, Ordering::Relaxed);
        pbar.inc(1);
    });
    let read_err = read_err_files.into_inner();
    let json_err = json_err_lines.into_inner();
    let hash_err = hash_err_lines.into_inner();
    if read_err + json_err + hash_err > 0 {
        println!(
            "Skipped: {} files with read errors, {} lines with JSON errors, {} lines with hash errors",
            read_err, json_err, hash_err
        );
    }

    let kept_docs = if let Some(_annokey) = annotate {
        counter.len()
    } else {
        kept_docs.into_inner()
    };
    Ok((seen_docs.into_inner(), kept_docs))
}

/// Builds a hash frequency counter by scanning a file.
///
/// Used in annotation mode to count total occurrences of each document
/// before writing output.
fn build_out_counter<K: DocHash>(
    p: &PathBuf,
    text_key: &String,
    hash_key: &Option<String>,
    counter: &DashMap<K, usize>,
) -> Result<(), Error> {
    let data = read_pathbuf(p, true).unwrap();
    for line in data.lines() {
        let line = line.unwrap();
        let line_json: Value = serde_json::from_str(&line).unwrap();
        let hash_val = get_hash_val::<K>(&line_json, text_key, hash_key).unwrap();
        counter.entry(hash_val).and_modify(|c| *c += 1).or_insert(1);
    }
    Ok(())
}

/// Processes a single file, removing or annotating duplicates.
///
/// # Arguments
///
/// * `p` - Path to input file
/// * `output_filename` - Path where output should be written
/// * `text_key` - JSON key containing document text
/// * `hash_key` - Optional pre-computed hash field
/// * `counter` - Thread-safe hash occurrence counter
/// * `annotate` - Optional annotation key (None = remove duplicates)
///
/// # Returns
///
/// Tuple of (documents_seen, documents_kept)
fn exact_dedup_file<K: DocHash>(
    p: PathBuf,
    output_filename: PathBuf,
    text_key: &String,
    hash_key: &Option<String>,
    counter: &DashMap<K, usize>,
    annotate: &Option<String>,
    read_err_files: &AtomicUsize,
    json_err_lines: &AtomicUsize,
    hash_err_lines: &AtomicUsize,
) -> Result<(usize, usize), Error> {
    let mut seen = 0;
    let mut kept = if let Some(_anno) = annotate {
        0 // counter.len()
    } else {
        0
    };

    let mut writer = create_writer(&output_filename).unwrap();
    let data = read_pathbuf(&p, true).unwrap();
    for line in data.lines() {
        // Tolerate truncated .jsonl.zst files (premature EOF mid-frame) and malformed
        // JSON lines: bump a counter and either stop reading this file (read error) or
        // skip the line (json/hash error). Aggregate counts are printed by the caller.
        let line = match line {
            Ok(l) => l,
            Err(_) => {
                read_err_files.fetch_add(1, Ordering::Relaxed);
                break;
            }
        };
        seen += 1;
        let mut line_json: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => {
                json_err_lines.fetch_add(1, Ordering::Relaxed);
                continue;
            }
        };
        let hash_val = match get_hash_val::<K>(&line_json, text_key, hash_key) {
            Ok(v) => v,
            Err(_) => {
                hash_err_lines.fetch_add(1, Ordering::Relaxed);
                continue;
            }
        };
        if let Some(annotate_key) = annotate {
            let count = counter.get(&hash_val).unwrap();
            let anno_data = json!({"hash": hash_val.to_json(),
			 					   "num_dups": *count});
            json_set(&mut line_json, &annotate_key, anno_data).unwrap();
            writer
                .write_line(&serde_json::to_vec(&line_json).unwrap())
                .unwrap();
        } else {
            let count = *counter.entry(hash_val).and_modify(|c| *c += 1).or_insert(1);
            if count == 1 {
                kept += 1;
                writer
                    .write_line(&serde_json::to_vec(&line_json).unwrap())
                    .unwrap();
            }
        }
    }
    writer.finish().unwrap();

    Ok((seen, kept))
}

/// Extracts or computes the hash value for a document.
///
/// If `hash_key` is provided, reads the pre-computed hash from that JSON field.
/// Otherwise, hashes the text content using the appropriate hash function.
fn get_hash_val<K: DocHash>(
    json_obj: &Value,
    text_key: &String,
    hash_key: &Option<String>,
) -> Result<K, Error> {
    let hash_val: K = if let Some(key) = hash_key {
        // type of this should be K
        let hash_val_value = json_get(json_obj, key).unwrap();
        DocHash::from_json(hash_val_value).unwrap()
    } else {
        let text = json_obj
            .get(text_key)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        DocHash::hash_string(&text)
    };
    Ok(hash_val)
}
