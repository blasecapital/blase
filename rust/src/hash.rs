// /src/hash.rs

use std::fs::File;
use std::io::{BufReader, Read, Result};
use sha2::{Sha256, Digest};

pub fn byte_hasher_be(input: &[u8]) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(input);
    let hash = hasher.finalize();
    Ok(hex::encode(hash))
}

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