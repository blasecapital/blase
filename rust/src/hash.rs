// /src/hash.rs

use std::fs::{File, metadata};
use std::io::{self, BufReader, Read, Result};
use sha2::{Sha256, Digest};
use std::path::Path;

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
/// - `path`: A path reference to the file (`impl AsRef<Path>`).
///
/// # Returns
/// A `Result<String>` containing the hex-encoded SHA-256 digest of the file’s contents,
/// or an `io::Error` if the path does not exist, is not a regular file, or cannot be read.
///
/// # Errors
/// - Returns `ErrorKind::NotFound` if the file does not exist.
/// - Returns `ErrorKind::InvalidInput` if the path is not a regular file (e.g. a directory).
/// - Returns `ErrorKind::PermissionDenied` if the file cannot be opened due to permissions.
/// - Propagates other I/O errors encountered during reading.
///
/// # Example
/// 
/// let hash = file_hasher_be("path/to/file.txt").unwrap();
/// println!("Hash: {}", hash);
/// assert_eq!(hash.len(), 64); // SHA-256 digest in hex
/// 
pub fn file_hasher_be<P: AsRef<Path>>(path: P) -> io::Result<String> {
    let p = path.as_ref();

    // Fail fast with a clear std::io::Error (no anyhow)
    let meta = metadata(p)?;
    if !meta.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("path is not a regular file: {}", p.display()),
        ));
    }

    let file = File::open(p)?;
    let mut reader = BufReader::new(file);
    let mut buf = [0u8; 8192];
    let mut hasher = Sha256::new();

    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};
    use std::env;

    fn uniq_tmp_dir() -> io::Result<std::path::PathBuf> {
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let dir = env::temp_dir().join(format!("blase_hash_test_{ts}"));
        fs::create_dir(&dir)?;
        Ok(dir)
    }

    #[test]
    fn byte_hasher_deterministic_and_hex_len() {
        let h1 = byte_hasher_be(b"abc").unwrap();
        let h2 = byte_hasher_be(b"abc").unwrap();
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
        assert!(h1.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn file_and_bytes_agree() -> io::Result<()> {
        let dir = uniq_tmp_dir()?;
        let file_path = dir.join("x.bin");
        let mut f = File::create(&file_path)?;
        let data = b"content-123";
        f.write_all(data)?;
        drop(f);

        let file_digest = file_hasher_be(&file_path)?;
        let bytes_digest = byte_hasher_be(data)?;
        assert_eq!(file_digest, bytes_digest);

        // cleanup
        fs::remove_file(&file_path).ok();
        fs::remove_dir(&dir).ok();
        Ok(())
    }

    #[test]
    fn non_file_path_errors() -> io::Result<()> {
        let dir = uniq_tmp_dir()?;
        let err = file_hasher_be(&dir).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        // cleanup
        fs::remove_dir(&dir).ok();
        Ok(())
    }
}