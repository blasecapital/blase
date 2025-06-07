// /src/lib.rs

// Maturin develop instructions:
//  1) conda init 
//  2) exec bash
//  3) conda activate base
//  4) cd ./rust 
//  5) maturin develop
//
// On local machine:
//  1) activate venv
//  2) pip install maturin
//  3) download rust
//  4) cd <rust directory>
//  5) maturin develop

use pyo3::prelude::*;

mod hash;

use hash::{byte_hasher_be, file_hasher_be};

#[pyfunction]
fn byte_hasher(input: &[u8]) -> PyResult<String> {
    Ok(byte_hasher_be(input)?)
}

#[pyfunction]
fn file_hasher(path: &str) -> PyResult<String> {
    Ok(file_hasher_be(path)?)
}

#[pymodule]
fn rust_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(byte_hasher, m)?)?;
    m.add_function(wrap_pyfunction!(file_hasher, m)?)?;
    Ok(())
}