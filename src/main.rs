//! # Document Deduplication CLI
//!
//! Command-line interface for exact and fuzzy (MinHash) document deduplication.
//!
//! ## Overview
//!
//! This tool supports four deduplication strategies:
//!
//! | Method | Storage | Best For |
//! |--------|---------|----------|
//! | Exact + Memory | In-memory | Small datasets, simple exact matching |
//! | Exact + Disk | Disk-based | Large datasets, exact matching, distributed |
//! | MinHash + Memory | In-memory | Small datasets, fuzzy matching |
//! | MinHash + Disk | Disk-based | Large datasets, fuzzy matching, distributed |
//!
//! ## Exact vs. Fuzzy Deduplication
//!
//! - **Exact**: Removes documents with identical content (or identical hash keys).
//!   Fast and deterministic, but misses near-duplicates.
//!
//! - **Fuzzy (MinHash)**: Uses locality-sensitive hashing to find near-duplicates
//!   based on Jaccard similarity. Follows the algorithm from
//!   [Lee et al. 2021](https://arxiv.org/abs/2107.06499).
//!
//! ## Memory vs. Disk
//!
//! - **Memory**: Stores intermediate data structures in RAM. Simpler to use,
//!   no setup required, but limited by available memory.
//!
//! - **Disk**: Stores intermediate files on disk. Supports datasets that don't
//!   fit in memory and enables distributed processing across multiple machines.
//!
//! ## Quick Start Examples
//!
//! ### Exact Deduplication (Small Dataset)
//! ```bash
//! cargo run --release -- exact-dedup-memory \
//!   --input-dir /data/docs \
//!   --output-dir /data/deduped \
//!   --text-key "content"
//! ```
//!
//! ### Fuzzy Deduplication (Small Dataset)
//! ```bash
//! cargo run --release -- minhash-memory \
//!   --input-dir /data/docs \
//!   --storage-dir /tmp/work \
//!   --output-dir /data/deduped \
//!   --text-key "text" \
//!   --num-buckets 20 \
//!   --bucket-size 5
//! ```
//!
//! ### Fuzzy Deduplication (Large Dataset, Distributed)
//! ```bash
//! # Step 1: Build file map (run once)
//! cargo run --release -- mh-build-file-map \
//!   --input-dir /data/docs \
//!   --storage-dir /shared/work
//!
//! # Step 2: Hash documents (run on multiple workers with different path-chunk values)
//! cargo run --release -- mh-hash-docs \
//!   --local-input /data/docs \
//!   --storage-dir /shared/work \
//!   --text-key "text" \
//!   --path-chunk 0 \
//!   --num-path-chunks 10
//!
//! # Step 3: Gather edges (run once, requires all signatures)
//! cargo run --release -- mh-gather-edges \
//!   --storage-dir /shared/work
//!
//! # Step 4: Build union-find (run once on single machine)
//! cargo run --release -- mh-build-uf \
//!   --storage-dir /shared/work \
//!   --num-path-chunks 10
//!
//! # Step 5: Clean files (run on multiple workers with different path-chunk values)
//! cargo run --release -- mh-clean-files \
//!   --input-dir /data/docs \
//!   --storage-dir /shared/work \
//!   --output-dir /data/deduped \
//!   --path-chunk 0 \
//!   --num-path-chunks 10 \
//!   --remove-duplicates true
//! ```

// External crates
use crate::sa_base::alternative_merge;
use clap::{Parser, Subcommand};

// Standard library
use std::option::Option;
use std::path::PathBuf;

// Internal crate imports
use crate::exact_dedup_disk::{exact_dedup_disk_group, exact_dedup_disk_prune};
use crate::exact_dedup_memory::exact_dedup_memory;
use crate::minhash_disk::{
    mh_build_file_map, mh_build_uf, mh_clean_files, mh_gather_edges, mh_hash_docs,
};
use crate::minhash_memory::minhash_memory;
use crate::sa_base::{get_matches_serial, make_sa_tables_cmd, merge_matches, sa_annotate_files, get_matches_parallel};
use crate::true_jaccard::true_jaccard;

pub mod exact_dedup_disk;
pub mod exact_dedup_memory;
pub mod minhash_base;
pub mod minhash_config;
pub mod minhash_disk;
pub mod minhash_memory;
pub mod sa_base;
pub mod storage;
pub mod true_jaccard;
pub mod uf_rush2;
pub mod utils;
// pub mod bitpack_vec;
pub mod table_old;
pub mod table_generic;
pub mod sa_utils;
pub mod sa_config;
pub mod compact_uint;
// pub mod table_new;
/* 4 basic use cases here:
{Exact, Fuzzy} x {Memory, Disk} deduplication:

Exact vs Fuzzy(minhash) Deduplication:
- Exact deduplication only keeps _one_ copy of a particular document, according to an either prespecified
  identifying field (i.e. a hash of the text), or will hash the text itself.
- Fuzzy deduplication using minhash performs minhash according to this schema: https://arxiv.org/abs/2107.06499
  More details on how this works in minhash.rs

Memory or Disk-based settings:
- In the memory-based settings, the necessary intermediate data structures are stored in memory. This use case
  is good for when the dataset you're deduplicating is relatively small. This should be plug-n-play and does not
  require careful system setup.
- In the disk-based setting: the intermediate data structures are stored on disk. This is useful for when the
  dataset is large and distributed processing is needed. See the README/tutorials for these large-scale jobs.
*/

/*
New plan:
It helps to consider minHash in phases:

Phase 1: Look at all the file paths we want to run minHash dedup on and rename them
         with usize identifiers. Store this in a FileMap structure.
         In this way, all documents are now uniquely identified by their
         (path_id, line_num)

Phase 2: Compute (band_id, signature, doc_id) for every document.
         This is the heavy minhashing step, but can be parallelized
         across chunks over the paths.

         Essentially this works as follows -- for each path
         we loop over documents and compute it's minhash signatures.
         We compute band_size * num_bands signatures, and then hash
         the signature across each band (rows in the pic below).
         If any two documents share the same (band_id, siganture), then
         they are duplicates.

                        Band Size
             ----------------------------
            |                            |
            |                            |
      Bands |                            |
            |                            |
            |                            |
            ------------------------------


Phase 3: Collect edges, linking all the documents that share the same
         (band_id, siganture) together. Store these in a separate set of files.

Phase 4: Combine all the edges and build a GLOBAL union find data structure and
         use this to collect connected components. Store both these connected
         components (for later examination), and collect the lines that should
         be removed from the dataset (grouped by path_chunk, so they can be pruned
         in a multinode setting)

Phase 5: Clean the duplicate documents from the paths. Phase 4 tells us which
         lines of each path should be removed.


(
Auxiliary phase: For examination purposes, we can look at each connected component
                  and get stats on the true pairwise jaccard similarity between
                  documents marked as "duplicates"
)

----------------
Some design notes:

+ Config: This can be done without a config, but your

+ Disk space: We rely heavily on storing auxiliary data structures on disk.
              Basically there's a state change after every phase where we make
              some new files after every phase. I'll build some python code
              to read these for debuggy purposes later (TBD). Each section
              will describe the file structure and contents in comments below.

+ Parallelism: Phase 3 and phase 4 need to be done globally (unfortunately,
               there's no avoiding at least one global step). Phase 1 only
               really needs access to the filenames but is very cheap.
               Phase 2, 4, and 5 are the heavy steps and can be done across
               file parallelism. This would require some coordination with the
               global data structures generated by phase 3 and 4.

+ s3: We don't touch s3 in rust. I have yet to find a package that works
      reliably, so let's just assume that all interaction between the LFS
      and s3 is done outside the context of this rust code


+ Input path structure: For simplicity, we require that the input paths all
                        live in one directory and have unique basenames.


+ One fell swoop: if your dataset is small enough that you just want to do
                  all phases in one command, use `min-hash`
*/

/*=================================================================
=                                  ARGS                           =
=================================================================*/

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct ArgParser {
    #[clap(subcommand)]
    command: Commands,

    #[arg(long, default_value_t = 0)]
    threads: usize,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /*============================================================
    =            Exact Deduplication Methods                     =
    ============================================================*/
    /// Exact deduplication for small datasets (all-in-memory processing)
    ///
    /// Removes documents with identical content in a single pass. All data
    /// is processed in memory, so this is best for datasets under ~10GB.
    ///
    /// EXAMPLE:
    ///   cargo run --release --  exact-dedup-memory \
    ///     --input-dir /data/documents \
    ///     --output-dir /data/unique \
    ///     --text-key "content" \
    ///     --annotate-key "duplicate_info"
    #[clap(arg_required_else_help = true)]
    ExactDedupMemory {
        /// Directory containing input JSONL files
        #[arg(required = true, long)]
        input_dir: PathBuf,

        /// Directory where deduplicated files will be written
        #[arg(required = true, long)]
        output_dir: PathBuf,

        /// JSON key containing document text
        #[arg(long, default_value_t=String::from("text"))]
        text_key: String,

        /// Optional: JSON key containing pre-computed document hash
        /// If not provided, text will be hashed automatically
        #[arg(long)]
        hash_key: Option<String>,

        /// Number of bits for document hash (if hash_key not provided)
        #[arg(long, default_value_t = 128)]
        hash_bits: usize,

        /// Optional: Add duplicate info to documents instead of removing them
        /// Can be nested, e.g., "metadata.duplicates"
        #[arg(long)]
        annotate_key: Option<String>,
    },

    /// Exact deduplication step 1/2: Group documents by hash
    ///
    /// First stage of disk-based exact deduplication for large datasets.
    /// Hashes all documents and groups them into bins for parallel processing.
    ///
    /// EXAMPLE:
    ///   cargo run --release --  exact-dedup-disk-group \
    ///     --input-dir /data/documents \
    ///     --storage-dir /scratch/work \
    ///     --hash-key "doc_id" \
    ///     --num-bins 100
    #[clap(arg_required_else_help = true)]
    ExactDedupDiskGroup {
        /// Directory containing input JSONL files
        #[arg(required = true, long)]
        input_dir: PathBuf,

        /// Working directory for intermediate files
        #[arg(required = true, long)]
        storage_dir: PathBuf,

        /// JSON key containing document text
        #[arg(long, default_value_t=String::from("text"))]
        text_key: String,

        /// JSON key where the hash of the data will live
        #[arg(long, required = true)]
        hash_key: String,

        /// Number of bits for document hash
        #[arg(long, default_value_t = 128)]
        hash_bits: usize,

        /// Number of bins to partition documents into
        /// More bins = better parallelism but more files
        #[arg(long, required = true)]
        num_bins: usize,
    },

    /// Exact deduplication step 2/2: Remove duplicates from groups
    ///
    /// Second stage of disk-based exact deduplication. Processes the grouped
    /// documents and removes duplicates.
    ///
    /// EXAMPLE:
    ///   cargo run --release --  exact-dedup-disk-prune \
    ///     --storage-dir /scratch/work \
    ///     --output-dir /data/unique \
    ///     --hash-key "doc_id"
    #[clap(arg_required_else_help = true)]
    ExactDedupDiskPrune {
        /// Working directory containing grouped documents
        #[arg(required = true, long)]
        storage_dir: PathBuf,

        /// Directory where deduplicated files will be written
        #[arg(required = true, long)]
        output_dir: PathBuf,

        /// JSON key used for deduplication (must match step 1)
        #[arg(required = true, long)]
        hash_key: String,

        /// Optional: Add duplicate info instead of removing
        #[arg(long)]
        annotate_key: Option<String>,
    },

    /*============================================================
    =            MinHash Deduplication Methods                   =
    ============================================================*/
    /// MinHash fuzzy deduplication for small datasets (all-in-memory)
    ///
    /// Removes near-duplicate documents using MinHash LSH. Runs entire
    /// pipeline in one command. Best for datasets under ~10GB.
    ///
    /// EXAMPLE:
    ///   cargo run --release --  minhash-memory \
    ///     --input-dir /data/documents \
    ///     --storage-dir /tmp/work \
    ///     --output-dir /data/deduped \
    ///     --text-key "content" \
    ///     --num-buckets 20 \
    ///     --bucket-size 5 \
    ///     --remove-duplicates true \
    ///     --cleanup-storage
    #[clap(arg_required_else_help = true)]
    MinhashMemory {
        /// Directory containing input JSONL files
        #[arg(required = true, long)]
        input_dir: PathBuf,

        /// Working directory for intermediate files
        #[arg(required = true, long)]
        storage_dir: PathBuf,

        /// Directory where deduplicated files will be written
        #[arg(required = true, long)]
        output_dir: PathBuf,

        /// JSON key containing document text
        #[arg(long, default_value_t=String::from("text"))]
        text_key: String,

        /// Optional: Path to YAML configuration file
        #[arg(long)]
        config: Option<PathBuf>,

        /// Number of LSH bands (more = stricter matching)
        #[arg(long)]
        num_buckets: Option<usize>,

        /// Hash values per band (more = stricter matching)
        #[arg(long)]
        bucket_size: Option<usize>,

        /// N-gram size for shingling (default: 5)
        #[arg(long)]
        ngram_size: Option<usize>,

        /// Random seed for reproducibility
        #[arg(long)]
        permutation_seed: Option<u64>,

        /// Number of signature chunks in the (temporary) storage
        #[arg(long)]
        num_sig_chunks: Option<usize>,

        /// Tokenizer: "cl100k", "p50k", "uniseg", or character-level (default)
        #[arg(long)]
        tokenizer: Option<String>,

        /// Add duplicate metadata to documents instead of removing
        #[arg(long)]
        annotate: Option<bool>,

        /// JSON key for annotations (e.g., "dedup_info")
        #[arg(long)]
        annotate_key: Option<String>,

        /// Delete input files after processing
        #[arg(long)]
        delete_while_cleaning: Option<bool>,

        /// Remove duplicate documents (keep first occurrence only)
        #[arg(long)]
        remove_duplicates: Option<bool>,

        /// Delete storage directory after completion
        #[arg(long, default_value_t = false)]
        cleanup_storage: bool,
    },

    /// MinHash step 1/5: Build file map
    ///
    /// Creates an index mapping file paths to integer IDs. Required before
    /// any other MinHash steps.
    ///
    /// EXAMPLE:
    ///   cargo run --release --  mh-build-file-map \
    ///     --input-dir /data/documents \
    ///     --storage-dir /shared/work
    #[clap(arg_required_else_help = true)]
    MhBuildFileMap {
        /// Directory containing input JSONL files
        #[arg(required = true, long)]
        input_dir: PathBuf,

        /// Working directory where file map will be saved
        #[arg(required = true, long)]
        storage_dir: PathBuf,
    },

    /// MinHash step 2/5: Compute signatures for document chunk
    ///
    /// Computes MinHash signatures for a subset of documents. Can be run
    /// in parallel across multiple machines with different path-chunk values.
    ///
    /// EXAMPLE:
    ///   # Worker 0 processes chunk 0 of 10
    ///   cargo run --release --  mh-hash-docs \
    ///     --local-input /data/documents \
    ///     --storage-dir /shared/work \
    ///     --text-key "text" \
    ///     --path-chunk 0 \
    ///     --num-path-chunks 10 \
    ///     --num-buckets 20 \
    ///     --bucket-size 5
    #[clap(arg_required_else_help = true)]
    MhHashDocs {
        /// Directory containing input files to process
        #[arg(required = true, long)]
        local_input: PathBuf,

        /// Working directory containing file map
        #[arg(required = true, long)]
        storage_dir: PathBuf,

        /// JSON key containing document text
        #[arg(long, default_value_t=String::from("text"))]
        text_key: String,

        /// Optional: Path to YAML configuration file
        #[arg(long)]
        config: Option<PathBuf>,

        /// This worker's chunk ID (0 to num-path-chunks - 1)
        #[arg(long, required = true)]
        path_chunk: usize,

        /// Total number of chunks for parallel processing
        #[arg(long, required = true)]
        num_path_chunks: usize,

        /// Number of LSH bands
        #[arg(long)]
        num_buckets: Option<usize>,

        /// Hash values per band
        #[arg(long)]
        bucket_size: Option<usize>,

        /// N-gram size for shingling
        #[arg(long)]
        ngram_size: Option<usize>,

        /// Random seed for reproducibility
        #[arg(long)]
        permutation_seed: Option<u64>,

        /// Tokenizer type
        #[arg(long)]
        tokenizer: Option<String>,

        /// Expected total number of documents (affects signature encoding)
        #[arg(long)]
        num_docs: Option<usize>,

        /// Maximum lines per file (affects line number encoding)
        #[arg(long)]
        max_lines_per_path: Option<usize>,

        /// Number of signature chunks for distribution
        #[arg(long)]
        num_sig_chunks: Option<usize>,
    },

    /// MinHash step 3/5: Gather edges from matching signatures
    ///
    /// Groups documents with identical signatures into duplicate candidates.
    /// All signature files must be present before running this step.
    ///
    /// EXAMPLE:
    ///   cargo run --release --  mh-gather-edges \
    ///     --storage-dir /shared/work
    #[clap(arg_required_else_help = true)]
    MhGatherEdges {
        /// Working directory containing signature files
        #[arg(required = true, long)]
        storage_dir: PathBuf,

        /// Optional: Path to YAML configuration file
        #[arg(long)]
        config: Option<PathBuf>,

        /// Expected total number of documents
        #[arg(long)]
        num_docs: Option<usize>,

        /// Maximum lines per file
        #[arg(long)]
        max_lines_per_path: Option<usize>,
    },

    /// MinHash step 4/5: Build Union-Find structure
    ///
    /// Identifies connected components of duplicates using Union-Find.
    /// WARNING: Must run on a single machine with access to all edge files.
    /// Cannot be parallelized across machines.
    ///
    /// EXAMPLE:
    ///   cargo run --release --  mh-build-uf \
    ///     --storage-dir /shared/work \
    ///     --num-path-chunks 10
    #[clap(arg_required_else_help = true)]
    MhBuildUf {
        /// Working directory containing edge files
        #[arg(required = true, long)]
        storage_dir: PathBuf,

        /// Optional: Path to YAML configuration file
        #[arg(long)]
        config: Option<PathBuf>,

        /// Number of chunks for cleaning metadata (should match step 5)
        #[arg(long, required = true)]
        num_path_chunks: usize,

        /// Maximum lines per file
        #[arg(long)]
        max_lines_per_path: Option<usize>,
    },

    /// MinHash step 5/5: Clean files using deduplication metadata
    ///
    /// Applies deduplication by removing or annotating duplicates. Can be run
    /// in parallel across multiple machines with different path-chunk values.
    ///
    /// EXAMPLE:
    ///   # Worker 0 processes chunk 0 of 10
    ///   cargo run --release --  mh-clean-files \
    ///     --input-dir /data/documents \
    ///     --storage-dir /shared/work \
    ///     --output-dir /data/deduped \
    ///     --path-chunk 0 \
    ///     --num-path-chunks 10 \
    ///     --remove-duplicates true \
    ///     --annotate true \
    ///     --annotate-key "dedup_info"
    #[clap(arg_required_else_help = true)]
    MhCleanFiles {
        /// Directory containing original input files
        #[arg(required = true, long)]
        input_dir: PathBuf,

        /// Working directory containing cleaning metadata
        #[arg(required = true, long)]
        storage_dir: PathBuf,

        /// Directory where cleaned files will be written
        #[arg(required = true, long)]
        output_dir: PathBuf,

        /// This worker's chunk ID (0 to num-path-chunks - 1)
        #[arg(required = true, long)]
        path_chunk: usize,

        /// Total number of chunks (must match step 4)
        #[arg(required = true, long)]
        num_path_chunks: usize,

        /// Optional: Path to YAML configuration file
        #[arg(long)]
        config: Option<PathBuf>,

        /// Add duplicate metadata to documents
        #[arg(long)]
        annotate: Option<bool>,

        /// JSON key for annotations
        #[arg(long)]
        annotate_key: Option<String>,

        /// Delete input files after processing
        #[arg(long)]
        delete_while_cleaning: Option<bool>,

        /// Remove duplicate documents
        #[arg(long)]
        remove_duplicates: Option<bool>,

        /// Delete storage directory after completion
        /// WARNING: Only set true on final worker to avoid race conditions
        #[arg(long, default_value_t = false)]
        cleanup_storage: bool,
    },

    #[clap(arg_required_else_help = true)]
    TrueJaccard {
        /// Directory containing annotated minhash data
        #[arg(required = true, long)]
        input_dir: PathBuf,

        /// Directory where output (annotated) files will be written
        #[arg(required = true, long)]
        output_dir: PathBuf,

        /// The key where the minhash group ids are contained (if missing, do nothing). If None, do full pairwise O(n^2) comparisons
        #[arg(long)]
        minhash_cc_id: Option<String>,

        /// If non-null, is a regex that, when selected for, creates groups of files to check (easier for parallelism)
        #[arg(long)]
        group_regex: Option<String>,

        /// Optional: Path to YAML configuration file
        #[arg(long)]
        config: Option<PathBuf>,

        /// Threshold for jaccard similarity,
        #[arg(required = true, long)]
        jaccard_threshold: f64,

        /// N-gram size for shingling (default: 5)
        #[arg(long)]
        ngram_size: Option<usize>,

        /// Tokenizer: "cl100k", "p50k", "uniseg", or character-level (default)
        #[arg(long)]
        tokenizer: Option<String>,

        /// Annotate keys for the output
        #[arg(required = true, long)]
        annotate_key: String,

        /// If the group is bigger than this size, just put these docs into the "hotnode_dir"
        #[arg(long)]
        hotnode_size: Option<usize>,

        /// If group is bigger than ^, this is where the docs go
        #[arg(long)]
        hotnode_dir: Option<PathBuf>,

        /// Parallel nest: How many outer loops we split this up into. Somewhere between 4-16 is probably best
        #[arg(long, default_value_t = 4)]
        parallel_nest: usize,

        /// Offset for the connected component id -- useful when doing this in a multi-node setting. Defaults to 0
        #[arg(long)]
        id_offset: Option<usize>,

        /// Delete input files after processing
        #[arg(long)]
        delete_while_cleaning: Option<bool>,
    },
    #[clap(arg_required_else_help = true)]
    SaMakeTables {
        #[arg(required = true, long)]
        input_dir: PathBuf,

        #[arg(required = true, long)]
        output_dir: PathBuf,

        #[arg(long)]
        config: Option<PathBuf>,

        #[arg(long)]
        file_map: Option<PathBuf>,

        #[arg(long)]
        tokenizer: Option<String>,

        #[arg(long)]
        max_lines_per_path: Option<usize>,

        #[arg(long)]
        text_key: Option<String>,

        #[arg(long, default_value_t=1.0)]
        num_batches: f32,
    },

    #[clap(arg_required_else_help = true)]
    SaGetMatchesSerial {
        #[arg(required = true, long)]
        storage_dir: PathBuf,

        #[arg(long, default_value_t = 500)]
        match_length: usize,
    },

    #[clap(arg_required_else_help = true)]
    SaGetMatchesParallel {
        #[arg(required = true, long)]
        storage_dir: PathBuf,

        #[arg(long, default_value_t = 500)]
        match_length: usize,
    },    

    #[clap(arg_required_else_help = true)]
    SaMergeMatches {
        #[arg(required = true, long)]
        storage_dir: PathBuf,
    },

    #[clap(arg_required_else_help = true)]
    SaAnnotate {
        #[arg(required = true, long)]
        input_dir: PathBuf,

        #[arg(required = true, long)]
        storage_dir: PathBuf,

        #[arg(required = true, long)]
        output_dir: PathBuf,

        #[arg(required = true, long)]
        annotate_key: String,

        #[arg(long)]
        text_key: Option<String>,
    },

    #[clap(arg_required_else_help = true)]
    SaAlt {
        #[arg(required = true, long)]
        text: PathBuf,

        #[arg(required = true, long)]
        offset: PathBuf,        

        #[arg(required = true, long)]
        sa_table: PathBuf,

        #[arg(long, default_value_t = 500)]
        match_length: usize,
    },
}

/*=================================================================
=                                 MAIN                            =
=================================================================*/

#[allow(unreachable_patterns)]
fn main() {
    let args = ArgParser::parse();
    let threads = args.threads;
    if threads != 0 {
        std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
    }

    let result = match &args.command {
        Commands::ExactDedupMemory {
            input_dir,
            output_dir,
            text_key,
            hash_key,
            hash_bits,
            annotate_key,
        } => exact_dedup_memory(
            input_dir,
            output_dir,
            text_key,
            hash_key.clone(),
            *hash_bits,
            annotate_key.clone(),
        ),

        Commands::ExactDedupDiskGroup {
            input_dir,
            storage_dir,
            text_key,
            hash_key,
            hash_bits,
            num_bins,
        } => exact_dedup_disk_group(
            input_dir,
            storage_dir,
            text_key,
            &hash_key.clone(),
            *hash_bits,
            *num_bins,
        ),

        Commands::ExactDedupDiskPrune {
            storage_dir,
            output_dir,
            hash_key,
            annotate_key,
        } => exact_dedup_disk_prune(storage_dir, output_dir, hash_key, annotate_key),

        Commands::MinhashMemory {
            input_dir,
            storage_dir,
            output_dir,
            text_key,
            config,
            num_buckets,
            bucket_size,
            ngram_size,
            permutation_seed,
            num_sig_chunks,
            tokenizer,
            annotate,
            annotate_key,
            delete_while_cleaning,
            remove_duplicates,
            cleanup_storage,
        } => minhash_memory(
            input_dir,
            storage_dir,
            output_dir,
            text_key,
            config,
            *num_buckets,
            *bucket_size,
            *ngram_size,
            *permutation_seed,
            num_sig_chunks.clone(),
            tokenizer.clone(),
            annotate.clone(),
            annotate_key.clone(),
            *delete_while_cleaning,
            *remove_duplicates,
            *cleanup_storage,
        ),

        Commands::MhBuildFileMap {
            input_dir,
            storage_dir,
        } => mh_build_file_map(input_dir, storage_dir),

        Commands::MhHashDocs {
            local_input,
            storage_dir,
            text_key,
            config,
            path_chunk,
            num_path_chunks,
            num_buckets,
            bucket_size,
            ngram_size,
            permutation_seed,
            tokenizer,
            num_docs,
            max_lines_per_path,
            num_sig_chunks,
        } => mh_hash_docs(
            local_input,
            storage_dir,
            text_key,
            config,
            *path_chunk,
            *num_path_chunks,
            *num_buckets,
            *bucket_size,
            *ngram_size,
            *permutation_seed,
            tokenizer.clone(),
            *num_docs,
            *max_lines_per_path,
            *num_sig_chunks,
        ),

        Commands::MhGatherEdges {
            storage_dir,
            config,
            num_docs,
            max_lines_per_path,
        } => mh_gather_edges(storage_dir, config, *num_docs, *max_lines_per_path),

        Commands::MhBuildUf {
            storage_dir,
            config,
            num_path_chunks,
            max_lines_per_path,
        } => mh_build_uf(
            storage_dir,
            config.clone(),
            *num_path_chunks,
            *max_lines_per_path,
        ),

        Commands::MhCleanFiles {
            input_dir,
            storage_dir,
            output_dir,
            path_chunk,
            num_path_chunks,
            config,
            annotate,
            annotate_key,
            delete_while_cleaning,
            remove_duplicates,
            cleanup_storage,
        } => mh_clean_files(
            input_dir,
            storage_dir,
            output_dir,
            *path_chunk,
            *num_path_chunks,
            config,
            *annotate,
            annotate_key.clone(),
            *delete_while_cleaning,
            *remove_duplicates,
            *cleanup_storage,
        ),

        Commands::TrueJaccard {
            input_dir,
            output_dir,
            minhash_cc_id,
            group_regex,
            config,
            jaccard_threshold,
            ngram_size,
            tokenizer,
            annotate_key,
            hotnode_size,
            hotnode_dir,
            parallel_nest,
            id_offset,
            delete_while_cleaning,
        } => true_jaccard(
            input_dir,
            output_dir,
            minhash_cc_id.clone(),
            group_regex.clone(),
            config.clone(),
            *jaccard_threshold,
            ngram_size.clone(),
            tokenizer.clone(),
            annotate_key,
            hotnode_size.clone(),
            hotnode_dir.clone(),
            *parallel_nest,
            id_offset.clone(),
            delete_while_cleaning.clone(),
        ),
        Commands::SaMakeTables {
            input_dir,
            output_dir,
            config,
            file_map,
            tokenizer,
            max_lines_per_path,
            text_key,
            num_batches
        } => make_sa_tables_cmd(
            input_dir,
            output_dir,
            config.clone(),
            file_map.clone(),
            tokenizer.clone(),
            max_lines_per_path.clone(),
            text_key.clone(),
            *num_batches
        ),

        Commands::SaGetMatchesSerial {
            storage_dir,
            match_length,
        } => get_matches_serial(storage_dir, *match_length),

        Commands::SaGetMatchesParallel {
            storage_dir,
            match_length,
        } => get_matches_parallel(storage_dir, *match_length),        

        Commands::SaMergeMatches {
            storage_dir,            
        } => merge_matches(storage_dir),

        Commands::SaAnnotate {
            input_dir,
            storage_dir,
            output_dir,
            annotate_key,
            text_key,
        } => sa_annotate_files(input_dir, storage_dir, output_dir, annotate_key, text_key.clone()),

        Commands::SaAlt {
            text,
            offset,
            sa_table,
            match_length,
        } => alternative_merge(text, offset, sa_table, *match_length),

        _ => Ok(()),
    };
    result.unwrap()
}
