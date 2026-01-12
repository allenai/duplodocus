use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use crate::minhash_base::FileMap;
use crate::utils::{json_set, json_get};
use ahash::HashMap;
use dashmap::DashMap;
use glob::glob;
use serde_json::json;
use serde_json::Value;
use std::cmp::Reverse;
use std::fs::File;
use std::io::BufRead;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Instant;

//use crate::table_old::SuffixTable;
use crate::table_generic::DynamicSuffixTable;
use anyhow::{Error, Result};
use gjson;
use mj_io::{
    build_pbar, get_output_filename, read_pathbuf, read_pathbuf_to_mem, write_mem_to_pathbuf,
};
use rayon;
use rayon::prelude::*;
use std::path::PathBuf;

use std::cmp::Ordering as StdOrdering;
use std::collections::BinaryHeap;

use crate::sa_utils::{FileRange, SAStream, TextIterator, MatchWriter, MatchWriterElement, TreeNode, LoserTree, read_u64_vec, sa_safety_check, calculate_bytes_per_chunk, ByteSize, get_byte_size};
use crate::sa_config::{SAConfigOverrides, SAConfig};
use crate::compact_uint::U40;
/*

Scaling notes:
- assume that all text can fit into main memory
- do NOT assume that all text + SA's can fit into main memory


Notes for mega-scale suffix-array usage:

Step 1: Make suffix array tables.
Output:
	- storage/file_map.json.gz	
	k separate files:
	- storage/text/text_part_XXXX.bin
	- storage/table/table_part_XXXX.bin
	- storage/offsets/offset_part_XXXX.bin

Where: 
	- text is a global concatenation of:
		<document_text>\xff\xff<path_id><line_num>
		where path_id and line_num are both little-ended u64's

	- table is the suffix table for the given text 
	- offsets is 
		<path_id><line_num><doc_len>
		where each are little-ended u64s

- So we:
1) load all text
2) from here, compute the SA sizes
	a) can do the 'large-as-possible', where we have 1 suffix array, but this is non-threaded
	b) can do the 'one SA per thread', where number of SA's are small (pay logK later)
	c) make chunks based on RAM sizes:
		each thread can have THREAD_MEM := SYS_MEM / THREAD_COUNT memory
		so that means each thread can have THREAD_MEM/9 * SAFETY_MARGIN text max
3) save everything


Step 2: 
Output: 
	- storage/matches/match_part_XXXX.bin
	  where the contents are 
	  <source><sa_value>
	  where each is a u64, and means that a repeat starts at text_part_<source>.bin[sa_value]


Step 3: just needs to load matches and offsets into main memory 

Step 4: standard annotation flow, shouldn't overrun memory

*/


/*======================================================
=                    MAKE TABLE                        =
======================================================*/
const MEMORY_SAFETY_MARGIN: f64 = 0.90;


pub fn make_sa_tables_cmd(
    input_dir: &PathBuf,
    output_dir: &PathBuf,
    config_opt: Option<PathBuf>,
    file_map_opt: Option<PathBuf>,
    tokenizer: Option<String>,
    max_lines_per_path: Option<usize>,
    text_key: Option<String>,
    num_batches: f32,
) -> Result<(), Error> {
    let file_map = if let Some(file_map_name) = file_map_opt {
        FileMap::load(&file_map_name).unwrap()
    } else {
        let file_map = FileMap::new(input_dir, &None).unwrap();
        file_map
            .save(&(output_dir.join("file_map.json.gz")))
            .unwrap();
        file_map
    };

    let config_overrides = SAConfigOverrides {
        tokenizer: tokenizer,
        max_lines_per_path: max_lines_per_path,
        text_key: text_key,
        random_seed: None,
    };

    let config_obj = SAConfig::load_with_overrides(config_opt, config_overrides).unwrap();

    make_sa_tables(&config_obj, &file_map, input_dir, output_dir, num_batches)
}

pub fn make_sa_tables(
    config_obj: &SAConfig,
    file_map: &FileMap,
    input_dir: &PathBuf,
    output_dir: &PathBuf,
    num_batches: f32
) -> Result<(), Error> {
    assert_eq!(
        1u16.to_ne_bytes(),
        1u16.to_le_bytes(),
        "System must be little-endian"
    );

    let start_main = Instant::now();
    println!("Building SA tables...");

    // Step 1: load all data into memory and then maybe tokenize if needed?
    let start_read = Instant::now();
    println!("Reading all documents texts...");
    let data_docs = load_data_docs(file_map, config_obj, input_dir).unwrap();
    println!(" {:?} DOCS TOTAL", data_docs.len());
    println!(" {:?} Bytes of text total", data_docs.par_iter().map(|(_, _, s)| s.len()).sum::<usize>());
    println!(
        "...Read all documents in {:?} secs",
        start_read.elapsed().as_secs()
    );

    // Safety check:
    let corpus_len: usize = data_docs.par_iter().map(|(_, _, doc)| doc.len() + 2 * 8).sum();
    // assert!(sa_safety_check(corpus_len));
    sa_safety_check(corpus_len);
    let thread_count = (rayon::current_num_threads() as f32 * num_batches) as usize;
    let chunk_text_byte_size = calculate_bytes_per_chunk(thread_count, MEMORY_SAFETY_MARGIN);




    // Step 2 (optional): Tokenize if needed
    // TODO: tokenizer thing


    // Step 3: Break into chunks and make SA table for each of them
    let owned_chunks: DashMap<usize, Vec<(usize, usize, String)>> = chunk_data_for_sa(config_obj, data_docs, chunk_text_byte_size, thread_count);

    // PRINT BLOCK
    let chunk_sizes: Vec<usize> = owned_chunks.par_iter().map(|entry| {
    	let v = entry.value();
    	v.iter().map(|(_, _, t)| t.len()).sum::<usize>()
    }).collect();

    println!("CHUNK BYTE LIMIT {:?} | CHUNK SIZES {:?}", chunk_text_byte_size, chunk_sizes); 
    let total_bytes = AtomicUsize::new(0);

    let total_pbar = build_pbar(owned_chunks.len() * 3, "Total steps");
    owned_chunks.into_par_iter().for_each(|(table_idx, chunk)| {
        let allocate_size = chunk
            .iter()
            .map(|(_, _, s)| s.len() + 2 + 2 * 8)
            .sum::<usize>();
        let mut document_offsets: Vec<u64> = Vec::new();
        document_offsets.extend(vec![u64::MAX, u64::MAX, 0]);
        let mut buf = Vec::with_capacity(allocate_size);

        chunk.into_iter().for_each(|(u1, u2, s)| {
            buf.extend_from_slice(s.as_bytes());
            buf.extend_from_slice(&[0xff, 0xff]);
            buf.extend_from_slice(&(u1 as u64).to_le_bytes());
            buf.extend_from_slice(&(u2 as u64).to_le_bytes());
            document_offsets.extend(vec![u1 as u64, u2 as u64, buf.len() as u64]);
            // Offsets are trips of (path_id, line_num, current_offset)
        });
        assert_eq!(document_offsets.len() % 3, 0);
        total_bytes.fetch_add(allocate_size, Ordering::SeqCst);
        total_pbar.inc(1);
        // And save concatenated text just for safekeeping
        let output_text_filename = output_dir
            .clone()
            .join("text")
            .join(format!("text_part_{:04}.bin", table_idx));
        write_mem_to_pathbuf(&buf, &output_text_filename).unwrap();
        total_pbar.inc(1);

        // And save the document offsets
        let output_offset_filename = output_dir
            .clone()
            .join("offsets")
            .join(format!("offset_part_{:04}.bin", table_idx));
        let offset_vec: &[u8] = bytemuck::cast_slice(&document_offsets);
        write_mem_to_pathbuf(offset_vec, &output_offset_filename).unwrap();

        // And then make the SA table too
        let table = DynamicSuffixTable::new(&buf, buf.len() as u64).unwrap();
        total_pbar.inc(1);
        let table_to_write: &[u8] = bytemuck::cast_slice(table.table());
        let output_table_filename = output_dir
            .clone()
            .join("table")
            .join(format!("table_part_{:04}.bin", table_idx));
        write_mem_to_pathbuf(table_to_write, &output_table_filename).unwrap();
    });

    println!(
        "Made {:?} tables of {:?} bytes total in {:?} seconds",
        thread_count,
        total_bytes.into_inner(),
        start_main.elapsed().as_secs()
    );
    Ok(())
}

pub fn load_data_docs(
    file_map: &FileMap,
    config: &SAConfig,
    local_input: &PathBuf,
) -> Result<Vec<(usize, usize, String)>, Error> {
    let text_key = config.text_key.clone();
    let mut extant_idxs: Vec<(&PathBuf, &usize)> = file_map
        .indices
        .par_iter()
        .filter(|(p, _idx)| local_input.clone().join(p).exists())
        .collect();
    extant_idxs.par_sort_unstable_by_key(|(_, &i)| i);

    let pbar = build_pbar(extant_idxs.len(), "Paths");
    let data_docs: Vec<(usize, usize, String)> = extant_idxs
        .par_iter()
        .flat_map(|(p, idx)| {
            let mut sub_docs: Vec<(usize, usize, String)> = Vec::new();
            let contents = read_pathbuf(&(local_input.clone().join(p)), true).unwrap();
            for (line_num, line) in contents.lines().enumerate() {
                let line = line.unwrap();
                let value = gjson::get(&line, &text_key).str().to_string();
                sub_docs.push((**idx, line_num, value));
            }
            pbar.inc(1);
            sub_docs
        })
        .collect();
    Ok(data_docs)
}

fn get_part_num(path: &PathBuf) -> Result<usize, Error> {
    Ok(path
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .split("_")
        .last()
        .unwrap()
        .parse::<usize>()
        .unwrap())
}

fn chunk_data_for_sa(
    config: &SAConfig, 
    data_docs: Vec<(usize, usize, String)>, 
    bytes_per_chunk: usize,
    thread_count: usize
) -> DashMap<usize, Vec<(usize, usize, String)>> {
    let owned_chunks: DashMap<usize, Vec<(usize, usize, String)>> = DashMap::new();
    let chunk_counter = AtomicUsize::new(0);
    // Step 1: shuffle and index the data
    let mut indexed: Vec<(u64, (usize, usize, String))> = data_docs
        .into_par_iter()
        .enumerate()
        .map(|(i, item)| {
            let mut rng = ChaCha8Rng::seed_from_u64(config.random_seed.wrapping_add(i as u64));
            (rng.gen(), item)
        })
        .collect();
    
    indexed.par_sort_unstable_by_key(|(key, _)| *key);

    // Step 2: distribute into thread buckets by MOVING (no clone!)
    let mut chunks: Vec<Vec<_>> = (0..thread_count).map(|_| Vec::new()).collect();
    
    for (i, item) in indexed.into_iter().enumerate() {
        chunks[i % thread_count].push(item);
    }
    
    // Step 3: split each bucket into memory-bounded chunks
    chunks.into_par_iter().for_each(|chunk| {
        let mut current_part = chunk_counter.fetch_add(1, Ordering::SeqCst);
        let mut current_size = 0;
        let mut current_chunk = Vec::new();

        for (_, (idx1, idx2, text)) in chunk {
            let item_size = text.len() + 16; // approximate overhead
            assert!(item_size < bytes_per_chunk, "Document too large for chunk limit!");
            
            if current_size + item_size > bytes_per_chunk && current_size > 0 {
                // Flush current chunk
                owned_chunks.insert(current_part, current_chunk);
                current_chunk = Vec::new();
                current_size = 0;
                current_part = chunk_counter.fetch_add(1, Ordering::SeqCst);
            }
            
            current_chunk.push((idx1, idx2, text));
            current_size += item_size;
        }
        
        // Don't forget the last chunk
        if !current_chunk.is_empty() {
            owned_chunks.insert(current_part, current_chunk);
        }
    });
    owned_chunks
}



/*============================================================
=                            PQ LOOPS                        =
============================================================*/

pub fn get_matches_serial(storage_dir: &PathBuf, match_length: usize) -> Result<(), Error> {
    /* Step 1:
    Load into memory:
        - file map
        - global offset file
        - all text files
        - all offset files
    And open stream readers from SA tables
    */
    let start_main = Instant::now();
    let text_lookup = make_text_lookups(storage_dir).unwrap();
    let offset_lookup = make_offset_lookups(storage_dir).unwrap();
    let table_streams = make_table_streams(storage_dir, &text_lookup, &offset_lookup).unwrap();

    let mut stream_map: HashMap<usize, SAStream<File>> = table_streams.into_iter().enumerate().map(|(i, stream)| (i, stream)).collect();    
    let match_writer = MatchWriter::new(&storage_dir.clone().join("matches")).unwrap();    
    let match_count = get_matches_parallel_thread(&mut stream_map, &match_writer, match_length, true, 0).unwrap();
    match_writer.finish().unwrap();
    println!(
        "Finished PQ in {:?} secs | found {:?} matches",
        start_main.elapsed().as_secs(),
        match_count,
    );
    Ok(())
}

fn make_text_lookups(storage_dir: &PathBuf) -> Result<Vec<Vec<u8>>, Error> {
    // Creates a lookup from idx -> text
    let text_pattern = storage_dir.clone().join("text").join("text_part_*");
    let text_objects: Vec<PathBuf> = glob(&text_pattern.to_str().unwrap())
        .unwrap()
        .into_iter()
        .map(|p| p.unwrap())
        .collect();
    let max_text_idx = AtomicUsize::new(0);
    let text_data: Vec<(usize, Vec<u8>)> = text_objects
        .into_par_iter()
        .map(|p| {
            //PAR
            let text_idx = get_part_num(&p).unwrap();
            max_text_idx.fetch_max(text_idx, Ordering::SeqCst);
            (
                text_idx,
                read_pathbuf_to_mem(&p).unwrap().into_inner().into_inner(),
            )
        })
        .collect();
    let mut text_lookup: Vec<Vec<u8>> = vec![Vec::new(); max_text_idx.into_inner() + 1];
    text_data
        .into_iter()
        .for_each(|(idx, v)| text_lookup[idx] = v);

    Ok(text_lookup)
}

fn make_offset_lookups(storage_dir: &PathBuf) -> Result<Vec<Vec<u64>>, Error> {
    // Creates a lookup from idx -> offset
    let offset_pattern = storage_dir.clone().join("offsets").join("offset_part_*");
    let offset_objects: Vec<PathBuf> = glob(&offset_pattern.to_str().unwrap())
        .unwrap()
        .into_iter()
        .map(|p| p.unwrap())
        .collect();
    let max_offset_idx = AtomicUsize::new(0);
    let offset_data: Vec<(usize, Vec<u64>)> = offset_objects
        .into_par_iter()
        .map(|p| {
            //PAR
            let offset_idx = get_part_num(&p).unwrap();
            max_offset_idx.fetch_max(offset_idx, Ordering::SeqCst);
            (
                offset_idx,
                read_u64_vec(&p).unwrap()
            )
        })
        .collect();
    let mut offset_lookup: Vec<Vec<u64>> = vec![Vec::new(); max_offset_idx.into_inner() + 1];
    offset_data
        .into_iter()
        .for_each(|(idx, v)| offset_lookup[idx] = v);

    Ok(offset_lookup)
}


fn make_table_streams<'stream, 'a>(
    storage_dir: &PathBuf,
    text_lookup: &'a Vec<Vec<u8>>,
    offset_lookup: &'a Vec<Vec<u64>>,
) -> Result<Vec<SAStream<'a, File>>, Error> {
    let table_pattern = storage_dir.clone().join("table").join("table_part_*");
    let table_objects: Vec<PathBuf> = glob(&table_pattern.to_str().unwrap())
        .unwrap()
        .into_iter()
        .map(|p| p.unwrap())
        .collect();

    let table_streams: Vec<SAStream<File>> = table_objects
        .into_par_iter()
        .map(|p| {
            let table_idx = get_part_num(&p).unwrap();
            let sa_element_size = get_byte_size(text_lookup[table_idx].len());
            let open_file = File::open(&p).unwrap();
            let stream = SAStream::new(
                open_file,
                text_lookup[table_idx].as_slice(),
                offset_lookup[table_idx].as_slice(),
                table_idx,
                16384,
                sa_element_size,
            )
            .unwrap();
            stream
        })
        .collect();

    Ok(table_streams)
}

pub fn get_matches_parallel(storage_dir: &PathBuf, match_length: usize) -> Result<(), Error> {
    /* Paralellism strategy : Paralellize across alphabet (based on statistics from one )
    1. Load all texts into memory
    2. Pick one suffix table and scan through it to get some reference suffices to serve as waypoints.
    3. Binary search through all suffix tables to break into start/end slices.
    4. Farm out:
        - Global lookup of all text data
        - Sub-streamers for each of the SA tables
        - Writers for each part
    */

    let start_main = Instant::now();
    println!("Starting match-gathering");
    let text_lookup = make_text_lookups(storage_dir).unwrap();
	let offset_lookup = make_offset_lookups(storage_dir).unwrap();    
    let thread_count = rayon::current_num_threads();
    // Get a #threads-1 list of internal boundaries to make #threads streams for each part
    let reference_suffices =
        get_reference_suffices(storage_dir, match_length, thread_count, &text_lookup).unwrap();
    // Get a #Parts x #threads streams
    let substreams =
        get_all_substreams(storage_dir, reference_suffices, match_length, &text_lookup, &offset_lookup).unwrap();

    // And then transpose into a #threads x #Parts grid
    let mut thread_iters: Vec<HashMap<usize, SAStream<FileRange>>> =
        (0..thread_count).map(|_| HashMap::default()).collect();
    substreams
        .into_iter()
        .enumerate()
        .for_each(|(part_num, part)| {
            part.into_iter()
                .enumerate()
                .for_each(|(thread_num, stream)| {
                    thread_iters[thread_num].insert(part_num, stream);
                })
        });


    // And set up the matches writers
    let match_writer = MatchWriter::new(&storage_dir.clone().join("matches")).unwrap();
    // Finally we can do the parallel merge thing:
    println!("Starting parallel match-finding...");
    let thread_pbar = build_pbar(thread_iters.len(), "Jobs");

    //pbar = build_pbar()
    thread_iters.par_iter_mut().enumerate().for_each(|(part_num, streams)| {
        get_matches_parallel_thread(streams, &match_writer, match_length, part_num==0, part_num).unwrap();
        thread_pbar.inc(1);
    });
    match_writer.finish().unwrap();


    println!("Found all matches in {:?} secs", start_main.elapsed().as_secs());
    Ok(())
}

fn get_reference_suffices(
    storage_dir: &PathBuf,
    match_length: usize,
    thread_count: usize,
    text_lookup: &Vec<Vec<u8>>,
) -> Result<Vec<Vec<u8>>, Error> {
    // Output is a vec of size (thread_count-1), with the internal boundaries needed to break SA's into thread_count parts
    // Get just one table and its reference text
    let table_pattern = storage_dir.clone().join("table").join("table_part_*");
    let table_objects: Vec<PathBuf> = glob(&table_pattern.to_str().unwrap())
        .unwrap()
        .into_iter()
        .map(|p| p.unwrap())
        .collect();
    let reference_table_path: PathBuf = table_objects.first().unwrap().clone();
    let mut reference_table_file: File = File::open(&reference_table_path).unwrap();
    let part_num = get_part_num(&reference_table_path).unwrap();
    let reference_text = text_lookup[part_num].as_slice();
    let sa_element_size = get_byte_size(reference_text.len());
    // And get the boundary indices
    let suffix_len = reference_table_file.metadata().unwrap().len() / sa_element_size as u64;
    assert!(thread_count > 1);
    assert!(suffix_len > thread_count.try_into().unwrap());
    let el_idxs: Vec<usize> = (1..thread_count)
        .map(|i| (i as f64 * ((suffix_len as f64) / (thread_count as f64))).round() as usize)
        .collect();

    // For boundary indices, get reference suffices:
    let mut reference_suffices: Vec<Vec<u8>> = Vec::new();
    el_idxs.into_iter().for_each(|i| {
        reference_table_file
            .seek(SeekFrom::Start((sa_element_size * i).try_into().unwrap()))
            .unwrap();
        let mut buffer = [0u8; 8];
        reference_table_file.read_exact(&mut buffer[..sa_element_size]).unwrap();
        let boundary_start = u64::from_le_bytes(buffer);
        let boundary_end = std::cmp::min(
            reference_text.len(),
            (boundary_start + match_length as u64).try_into().unwrap(),
        );
        let boundary: Vec<u8> =
            reference_text[boundary_start as usize..boundary_end as usize].to_vec();
        reference_suffices.push(boundary);
    });

    Ok(reference_suffices)
}

fn get_all_substreams<'a>(
    storage_dir: &PathBuf,
    boundaries: Vec<Vec<u8>>,
    match_length: usize,
    text_lookup: &'a Vec<Vec<u8>>,
    offset_lookup: &'a Vec<Vec<u64>>,
) -> Result<Vec<Vec<SAStream<'a, FileRange>>>, Error> {
    // Creates |thread_count| TextIterators for each SA Table, each offset by a little bit on each end from the boundaries

    // Step 1: Get table/text pairs
    let mut table_text_offset_trips: Vec<(PathBuf, &Vec<u8>, &Vec<u64>)> = Vec::new();
    let table_pattern = storage_dir.clone().join("table").join("table_part_*");
    let table_objects: Vec<PathBuf> = glob(&table_pattern.to_str().unwrap())
        .unwrap()
        .into_iter()
        .map(|p| p.unwrap())
        .collect();
    table_objects.into_iter().for_each(|p| {
        let part_num = get_part_num(&p).unwrap();
        table_text_offset_trips.push((p, &text_lookup[part_num], &offset_lookup[part_num]));
    });

    let output: Vec<Vec<SAStream<FileRange>>> = table_text_offset_trips
        .into_par_iter()
        .map(|(table_path, text, offset)| {
            get_substreams_from_boundaries(&table_path, text, offset, &boundaries, match_length).unwrap()
        })
        .collect();

    Ok(output)
}

fn get_substreams_from_boundaries<'a>(
    table_path: &PathBuf,
    text: &'a Vec<u8>,
    offset: &'a Vec<u64>,
    boundaries: &Vec<Vec<u8>>,
    match_length: usize,
) -> Result<Vec<SAStream<'a, FileRange>>, Error> {
    let mut table_file = File::open(table_path).unwrap();
    let sa_element_size = get_byte_size(text.len());
    let sa_len = table_file.metadata().unwrap().len() / sa_element_size as u64;

    let mut start_end_idxs: Vec<(usize, usize)> = Vec::new();
    // 0'th element

    start_end_idxs.push((
        0,
        sa_disk_bisect(
            &mut table_file,
            text,
            boundaries.first().unwrap(),
            match_length,
            true,
        )
        .unwrap(),
    ));
    for idx in 0..boundaries.len() - 1 {
        let l_boundary = &boundaries[idx];
        let r_boundary = &boundaries[idx + 1];
        let left_sa_index =
            sa_disk_bisect(&mut table_file, text, &l_boundary, match_length, false).unwrap();
        let right_sa_index =
            sa_disk_bisect(&mut table_file, text, &r_boundary, match_length, true).unwrap();
        start_end_idxs.push((left_sa_index, right_sa_index));
    }
    start_end_idxs.push((
        sa_disk_bisect(
            &mut table_file,
            text,
            boundaries.last().unwrap(),
            match_length,
            false,
        )
        .unwrap(),
        sa_len.try_into().unwrap(),
    ));

    // Dilate by 1 in each direction to ensure some overlap so we don't miss any intra-table duplicates
    let dilated_idxs: Vec<(usize, usize)> = start_end_idxs
        .into_iter()
        .map(|(l, r)| {
            let new_l = if l > 0 { l - 1 } else { l };

            let new_r = if r < sa_len.try_into().unwrap() {
                r + 1
            } else {
                r
            };
            (new_l, new_r)
        })
        .collect();

    // Then create FileRanges and SAStreams and TextIterators
    let text_slice = text.as_slice();
    let offset_slice = offset.as_slice();
    let part_num = get_part_num(&table_path).unwrap();
    let sa_streams: Vec<SAStream<FileRange>> = dilated_idxs
        .into_iter()
        .map(|(l, r)| {
            let new_file = File::open(table_path).unwrap();
            let file_range = FileRange::new(
                new_file,
                (l * sa_element_size).try_into().unwrap(),
                (r * sa_element_size).try_into().unwrap(),
            )
            .unwrap();
            let sas = SAStream::new(file_range, text_slice, offset_slice, part_num, 16384, sa_element_size).unwrap();
            sas
        })
        .collect();

    Ok(sa_streams)
}

fn sa_disk_bisect(
    table_file: &mut File,
    text: &Vec<u8>,
    boundary_str: &Vec<u8>,
    match_length: usize,
    find_right: bool, // true for bisect_right, false for bisect_left
) -> Result<usize, Error> {
    // Finds the index in the SA (in terms of SA ELEMENTS, not BYTES) such that everything to the left (idx <) is strictly less than boundary_str
    let sa_element_size = get_byte_size(text.len());
    let sa_len = (table_file.metadata().unwrap().len() / sa_element_size as u64) as usize;

    let mut left = 0;
    let mut right = sa_len;

    while left < right {
        let mid = left + (right - left) / 2;
        let mut buffer = [0u8; 8];
        table_file.seek(SeekFrom::Start((mid * sa_element_size) as u64))?;
        table_file.read_exact(&mut buffer[..sa_element_size])?;
        let t_start = u64::from_le_bytes(buffer) as usize;

        let t_end = std::cmp::min(text.len(), t_start + match_length);
        let t = &text[t_start..t_end];

        let cmp_len = std::cmp::min(boundary_str.len(), t.len());
        let comparison = boundary_str[..cmp_len].cmp(&t[..cmp_len]);

        let ordering = if comparison == StdOrdering::Equal {
            boundary_str.len().cmp(&t.len())
        } else {
            comparison
        };

        // bisect_left: find first where suffix >= boundary_str
        // bisect_right: find first where suffix > boundary_str
        let should_search_right = if find_right {
            ordering != StdOrdering::Less // boundary_str >= suffix
        } else {
            ordering == StdOrdering::Greater // boundary_str > suffix
        };

        if should_search_right {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    Ok(left)
}

fn get_matches_parallel_thread<'a, R: Read + ByteSize>(
    streams: &'a mut HashMap<usize, SAStream<'a, R>>,
    match_writer: &MatchWriter,
    match_length: usize,
    use_pbar: bool,
    part_num: usize,
) -> Result<usize, Error> {

    // Build pbar if requested
    let pbar_opt = if use_pbar {
    	let total_size: usize = streams.iter().map(|(_k,v)| v.byte_size as usize / v.element_size).sum::<usize>();
    	let pbar = build_pbar(total_size, "Thread bytes");
    	Some(pbar)
    } else {
    	None
    };

    // Convert streams to iterators
    let stream_iterators: HashMap<usize, TextIterator<R>> = streams
        .iter_mut()
        .map(|(k, v)| (*k, v.text_iter(match_length)))
        .collect();


    // Init LoserTree
    let mut match_count = 0;
    let part_num64 = part_num as u64;
    let mut loser_tree = LoserTree::new(stream_iterators, pbar_opt);
    let prev_min = loser_tree.pop().unwrap();
    if prev_min == None {
        return Ok(match_count);
    }
    let mut prev_min = prev_min.unwrap();
    let mut prev_lcp: Option<u64> = None;
    let mut currently_in_a_run = false;
    let mut prev_char_same = false;
    /* Loop through LoserTree 
    Steps: 
        1. Get the top value and if it shares match_length bytes w/ the last popped el, proceed to step 2. O/w loop.
        2a. If "not in a run", then the stashed "prev" needs to be written with all data and _current_ LCP 
        2b. IF "yes in a run", the check if _current_ LCP is > the prev LCP
    */
    while !loser_tree.peek().is_none() {
        let cur_min = loser_tree.pop_check().unwrap().unwrap();
        if cur_min.cmp_eq(&prev_min) {// If top value shares match_length bytes w/ previously popped el...
            let cur_lcp = cur_min.lcp(&prev_min, match_length);
            prev_char_same = cur_min.prev_char_same(&prev_min);
            if !prev_char_same { // If we would fully subsume this by some other thing
                let lcp_to_write = if let Some(prev_lcp_val) = prev_lcp {
                    prev_lcp_val.max(cur_lcp)
                } else {
                    cur_lcp
                };

                let prev_write_element = MatchWriterElement {
                    source: prev_min.source,
                    part_num: part_num64,
                    sa_idx: prev_min.sa_idx,
                    sa_value: prev_min.sa_value,
                    lcp: lcp_to_write,
                    first_run_el: !currently_in_a_run
                };
                match_writer.write_element(prev_write_element).unwrap();
                match_count += 1;
            }

            prev_lcp = Some(cur_lcp);
            currently_in_a_run = true;
        } else {
            if currently_in_a_run && !prev_char_same {// If I'm ending a run, write the previous thing in 
                let prev_write_element = MatchWriterElement {
                    source: prev_min.source,
                    part_num: part_num64,
                    sa_idx: prev_min.sa_idx,
                    sa_value: prev_min.sa_value,
                    lcp: prev_lcp.unwrap(),
                    first_run_el: false
                };                
                match_writer.write_element(prev_write_element).unwrap();
                match_count += 1;
            }
            prev_lcp = None;
            currently_in_a_run = false;            
            //prev_char_same = false;
        }
        prev_min = cur_min;
    }

    if currently_in_a_run && !prev_char_same {
        let prev_write_element = MatchWriterElement {
            source: prev_min.source,
            part_num: part_num64,
            sa_idx: prev_min.sa_idx,
            sa_value: prev_min.sa_value,
            lcp: prev_lcp.unwrap(),
            first_run_el: false
        };                
        match_writer.write_element(prev_write_element).unwrap();        
        match_count += 1;

    }


    Ok(match_count)
}




/*=====================================================================
=                            ALTERNATIVE FLOW                         =
=====================================================================*/

pub fn alternative_merge(
    text_loc: &PathBuf,
    offset_loc: &PathBuf,
    sa_table_loc: &PathBuf,
    match_length: usize,
) -> Result<(), Error> {
    let start_main = Instant::now();

    println!("Starting alt flow");
    let text: Vec<u8> = read_pathbuf_to_mem(text_loc)
        .unwrap()
        .into_inner()
        .into_inner();
    let sa_element_size = get_byte_size(text.len());
    let text_slice: &[u8] = text.as_slice();

    let offset = read_u64_vec(offset_loc).unwrap();   
    let offset_slice: &[u64] = offset.as_slice();    
    //let sa_table: Vec<u64> = read_u64_vec(sa_table_loc).unwrap();
    //let table_len = sa_table.len();
    let mut matches: Vec<u64> = Vec::new();
    let open_sa = File::open(sa_table_loc).unwrap();
    let mut stream = SAStream::new(open_sa, text_slice, offset_slice, 0, 16384, sa_element_size).unwrap();
    let mut text_iterator = stream.text_iter(match_length);
    let prev_min = text_iterator.next();
    if prev_min.is_none() {
        return Ok(());
    }
    let mut prev_min = prev_min.unwrap().unwrap();
    let mut currently_in_a_run = false;
    for el in text_iterator {
        let cur_node = el.unwrap();
        if cur_node.cmp_eq(&prev_min) {
            if currently_in_a_run {
                matches.push(prev_min.sa_value);
            }
            matches.push(cur_node.sa_value);
            currently_in_a_run = true;
        } else {
            currently_in_a_run = false;
        }
        prev_min = cur_node;
    }

    println!(
        "Finished alt flow in {:?} seconds | found {:?} matches",
        start_main.elapsed().as_secs(),
        matches.len()
    );
    Ok(())
}

/*=====================================================================
=                           MERGE MATCHES                             =
=====================================================================*/

pub fn merge_matches(storage_dir: &PathBuf) -> Result<(), Error> {
    let start_main = Instant::now();
    println!("Starting merging of matches");

    let input_match_dir = storage_dir.clone().join("matches");
    let offset_dir = storage_dir.clone().join("offsets");
    let offset_pattern = offset_dir.clone().join("offset_part_*.bin");

    let match_pattern = input_match_dir.clone().join("match_part*.bin");
    let merged_match_dir = storage_dir.clone().join("merged_matches");

    let matches: DashMap<usize, PathBuf> = glob(&match_pattern.to_str().unwrap())
        .unwrap()
        .into_iter()
        .map(|p| {
            let path = p.unwrap();
            let part_num = get_part_num(&path).unwrap();
            (part_num, path)
        })
        .collect();

    let offsets: DashMap<usize, PathBuf> = glob(&offset_pattern.to_str().unwrap())
        .unwrap()
        .into_iter()
        .map(|p| {
            let path = p.unwrap();
            let part_num = get_part_num(&path).unwrap();
            (part_num, path)
        })
        .collect();

    let pbar_counter = AtomicUsize::new(0);

    matches.into_par_iter().for_each(|(part_num, match_path)| {
        let offset_path = offsets.get(&part_num).unwrap();
        merge_match_path(
            match_path,
            &offset_path,
            &merged_match_dir,
            pbar_counter.fetch_add(1, Ordering::SeqCst) == 0,
        )
        .unwrap();
    });

    println!(
        "Merged all matches in {:?} secs",
        start_main.elapsed().as_secs()
    );
    Ok(())
}

pub fn merge_match_path(
    match_path: PathBuf,
    offset_path: &PathBuf,
    merged_match_dir: &PathBuf,
    pbar_flag: bool,
) -> Result<(), Error> {
    let part_num = get_part_num(&match_path).unwrap();
    let match_contents = read_pathbuf_to_mem(&match_path).unwrap().into_inner().into_inner();
    let elements = match_contents.chunks(MatchWriterElement::MATCH_EL_SIZE).map(|chunk| MatchWriterElement::from_bytes(part_num, chunk)).collect::<Vec<MatchWriterElement>>();

    let offset_trips = read_u64_vec(&offset_path).unwrap();
    let offset_trips: Vec<(u64, u64, u64)> = offset_trips
        .chunks_exact(3)
        .map(|c| (c[0], c[1], c[2]))
        .collect(); // (path_id, line_num, idx-ceiling)

    let pbar_opt = if pbar_flag {
        Some(build_pbar(elements.len() * 2, "Matches"))
    } else {
        None
    };

    let mut grouped_matches: HashMap<(usize, usize), Vec<MatchWriterElement>> = HashMap::default(); // maps (path_id, line_num) to modified matchWriter els
    elements.into_iter().for_each(|mut el| {
        let offset_index = offset_trips.partition_point(|&(_, _, document_end)| document_end <= el.sa_value);

        let offset_trip = offset_trips.get(offset_index).unwrap();
        let prev_offset_start = offset_trips.get(offset_index - 1).unwrap().2;
        el.sa_value -= prev_offset_start;
        let path_id = offset_trip.0;
        let line_num = offset_trip.1;        
        grouped_matches.entry(
            (path_id.try_into().unwrap(), line_num.try_into().unwrap())
        ).or_default().push(el);

        if let Some(ref pbar) = pbar_opt {
            pbar.inc(1);
        }
    });

    

    let mut output_bytes: Vec<u8> = Vec::new();

    grouped_matches.iter_mut().for_each(|(k, v)| { 
        let vlen = v.len();
        v.sort_by(|a, b| {a.sa_value.cmp(&b.sa_value)});
        let (safe_intervals, to_remove_intervals): (Vec<_>, Vec<_>) = v.into_iter().partition(|el| el.first_run_el);


        let remove_merged = merge_intervals(to_remove_intervals);
        let safe_merged = merge_intervals(safe_intervals);

        // and then remove any safe intervals        
        let output_intervals = subtract_intervals(remove_merged, safe_merged);        

        // And write into outputs 
        let path_id_bytes = k.0.to_le_bytes();
        let line_num_bytes = k.1.to_le_bytes();
        for (start, end) in output_intervals {
            output_bytes.extend(path_id_bytes);
            output_bytes.extend(line_num_bytes);
            output_bytes.extend(start.to_le_bytes());
            output_bytes.extend(end.to_le_bytes());
        }

        if let Some(ref pbar) = pbar_opt {
            pbar.inc(vlen.try_into().unwrap());
        }        
    });


    let file_stem = match_path.file_stem().unwrap().to_str().unwrap();
    let output_filename = merged_match_dir
        .clone()
        .join(format!("merged_{}.bin", file_stem));
    write_mem_to_pathbuf(&output_bytes, &output_filename)    
}

pub fn merge_intervals(els: Vec<&mut MatchWriterElement>) -> Vec<(u64, u64)> {
    if els.is_empty() {
        return Vec::new()
    }
    // Assume els is sorted to begin with
    let mut merged: Vec<(u64, u64)> = Vec::new();
    let first = els.first().unwrap();
    let mut start = first.sa_value;
    let mut cur_end = start + first.lcp;

    els.into_iter().skip(1).for_each(|el| {
        if el.sa_value > cur_end {
            merged.push((start, cur_end));
            start = el.sa_value;
        } else {}
        cur_end = el.sa_value + el.lcp
    });
    merged.push((start, cur_end));
    merged
}

fn subtract_intervals(vec_a: Vec<(u64, u64)>, vec_b: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    let mut result = Vec::new();
    let mut b_idx = 0;    
    for (a_start, a_end) in vec_a {
        let mut current_start = a_start;        
        // Skip intervals in vec_b that end before current interval starts
        while b_idx < vec_b.len() && vec_b[b_idx].1 <= current_start {
            b_idx += 1;
        }        
        // Process all intervals in vec_b that might overlap with current interval
        let mut temp_b_idx = b_idx;
        while temp_b_idx < vec_b.len() && vec_b[temp_b_idx].0 < a_end {
            let (b_start, b_end) = vec_b[temp_b_idx];            
            // If there's a gap before this b interval, add it to result
            if current_start < b_start.min(a_end) {
                result.push((current_start, b_start.min(a_end)));
            }            
            // Move current_start past this b interval
            current_start = b_end.max(current_start);            
            // If we've consumed the entire a interval, break
            if current_start >= a_end {
                break;
            }            
            temp_b_idx += 1;
        }
        
        // Add any remaining part of the a interval
        if current_start < a_end {
            result.push((current_start, a_end));
        }
    }
    
    result
}





/*======================================================================
=                            ANNOTATE FILES                            =
======================================================================*/

pub fn sa_annotate_files(
    input_dir: &PathBuf,
    storage_dir: &PathBuf,
    output_dir: &PathBuf,
    annotate_key: &String,
    text_key: Option<String>,
) -> Result<(), Error> {
    let start_main = Instant::now();
    println!("Starting annotation...");
    let text_key = text_key.unwrap_or(String::from("text"));

    let file_map_loc = storage_dir.clone().join("file_map.json.gz");
    let file_map = FileMap::load(&file_map_loc).unwrap();

    // Then loop through paths
    let merge_match_pattern = storage_dir
        .clone()
        .join("merged_matches")
        .join("merged_*.bin");
    let merge_match_objects: Vec<PathBuf> = glob(&merge_match_pattern.to_str().unwrap())
        .unwrap()
        .into_iter()
        .map(|p| p.unwrap())
        .collect();
    let anno_groups: DashMap<usize, DashMap<usize, Vec<(u64, u64)>>> = DashMap::new();


    merge_match_objects.into_par_iter().for_each(|p| {
        let contents: Vec<u64> = read_u64_vec(&p).unwrap();
        let contents: Vec<(u64, u64, u64, u64)> = contents
            .chunks_exact(4)
            .map(|c| (c[0], c[1], c[2], c[3]))
            .collect();
        contents
            .into_iter()
            .for_each(|(path_id, line_num, start, end)| {
                anno_groups
                    .entry(path_id as usize)
                    .or_default()
                    .entry(line_num as usize)
                    .or_default()
                    .push((start, end));
            });
    });

    let extant_idxs: Vec<(&PathBuf, &usize)> = file_map
        .indices
        .par_iter()
        .filter(|(p, _idx)| input_dir.clone().join(p).exists())
        .collect();
    let pbar = build_pbar(extant_idxs.len(), "Paths");

    let anno_docs = AtomicUsize::new(0);
    let anno_bytes = AtomicUsize::new(0);

    extant_idxs.into_par_iter().for_each(|(p, idx)| {
        let anno_group = anno_groups.entry(*idx).or_default();
        let input_path = input_dir.clone().join(&p);
        let output_path = get_output_filename(&input_path, input_dir, output_dir).unwrap();
        let (p_anno_docs, p_anno_bytes) = sa_annotate_path(input_path, &anno_group, &output_path, annotate_key, &text_key).unwrap();
        anno_docs.fetch_add(p_anno_docs, Ordering::SeqCst);
        anno_bytes.fetch_add(p_anno_bytes, Ordering::SeqCst);
        pbar.inc(1);
    });
    println!(
        "Annotated {:?} docs, marked {:?} bytes in {:?} secs",
		anno_docs.into_inner(),
		anno_bytes.into_inner(),        
        start_main.elapsed().as_secs()
    );
    Ok(())
}

fn sa_annotate_path(
    input_path: PathBuf,
    anno_group: &DashMap<usize, Vec<(u64, u64)>>,
    output_path: &PathBuf,
    annotate_key: &String,
    text_key: &String,
) -> Result<(usize, usize), Error> {
    let contents = read_pathbuf_to_mem(&input_path).unwrap();
    let mut output_contents: Vec<u8> = Vec::new();
    let mut p_anno_docs = 0;
    let mut p_anno_bytes = 0;
    for (line_num, line) in contents.lines().enumerate() {        
        let line = line.unwrap();

        let anno_val = anno_group.entry(line_num).or_default();
        if anno_val.len() == 0 {
            output_contents.extend(line.as_bytes());
        } else {                    
        	p_anno_docs += 1;
            let mut line_json: Value = serde_json::from_str(&line).unwrap();
            let text_len = json_get(&line_json, text_key).unwrap().as_str().unwrap().len() as u64;

            let anno_val: Vec<(u64, u64)> = anno_val.iter().filter_map(|(l, r)| {
                let l2 = l.min(&text_len);
                let r2 = r.min(&text_len);
                if l2 == r2 {
                    None
                } else {
                    Some((*l2, *r2))
                }
            }).collect();

            p_anno_bytes += anno_val.iter().map(|(l, r)| (r - l) as usize).sum::<usize>();
            let anno_val: Value = json!(*anno_val);
            json_set(&mut line_json, annotate_key, anno_val).unwrap();
            let output_line = serde_json::to_vec(&line_json).unwrap();
            output_contents.extend(output_line);
        }
        output_contents.push(b'\n')
    }

    write_mem_to_pathbuf(&output_contents, output_path).unwrap();
    Ok((p_anno_docs, p_anno_bytes))
}

