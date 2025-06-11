// /src/hash.rs

use std::fs::File;
use std::io::{BufReader, Read, Result};
use sha2::{Sha256, Digest};

/// Computes the SHA-256 hash of a byte slice and returns it as a hex string.
///
/// # Parameters
/// - `input`: A byte slice (`&[u8]`) to be hashed.
///
/// # Returns
/// A `Result<String>` containing the hex-encoded SHA-256 hash of the input bytes,
/// or an I/O error if the operation fails.
///
/// # Example
/// 
/// let hash = byte_hasher_be(b"hello world").unwrap();
/// assert_eq!(hash.len(), 64);
/// 
pub fn byte_hasher_be(input: &[u8]) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(input);
    let hash = hasher.finalize();
    Ok(hex::encode(hash))
}

/// Computes the SHA-256 hash of a file's contents and returns it as a hex string.
///
/// # Parameters
/// - `path`: A string slice that holds the path to the file.
///
/// # Returns
/// A `Result<String>` containing the hex-encoded SHA-256 hash of the file’s contents,
/// or an I/O error if the file can't be read.
///
/// # Example
/// 
/// let hash = file_hasher_be("path/to/file.txt").unwrap();
/// println!("Hash: {}", hash);
/// 
pub fn file_hasher_be(path: &str) -> Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut buffer = [0u8; 8192]; // 8 KB buffer
    let mut hasher = Sha256::new();

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    
    let hash = hasher.finalize();
    Ok(hex::encode(hash))
}