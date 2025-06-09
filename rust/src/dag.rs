// /rust/dag.rs

//! Scope
//! Gather data from step log folder
//! Create or update graph

use std::fs;
use std::collections::HashMap;
use serde_json::Value;
use serde::{Deserialize, Serialize};

// -------------- step_reader ---------------
pub fn step_reader(path: &str) -> Result<Value, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    let v: Value = serde_json::from_str(&contents)?;
    Ok(v)
}

// -------------- Step ---------------
#[derive(Deserialize, Debug)]
pub struct Step {
    function: String,
    inputs: Option<Value>,
    outputs: Option<Value>,
    params: Option<Value>,
    status: String,
    step_id: String,
    timestamp_end: String,
    timestamp_start: String,
}

pub fn to_step(step: Value) -> Step {
    let new_step: Step = serde_json::from_value(step).expect("Invalid step JSON");
    new_step
}

// -------------- walk_dir ---------------

pub fn walk_step_dir(dir: &str) -> Result<HashMap<String, Step>, Box<dyn std::error::Error>> {
    let mut step_map = HashMap::new();

    for entry_result in fs::read_dir(dir)?{
        let entry = entry_result?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("blase") {
            if let Some(path_str) = path.to_str() {
                let json_val = step_reader(path_str)?;
                let step = to_step(json_val);
                step_map.insert(step.step_id.clone(), step);
            } else {
                eprintln!("Warning: Skipping non-UTF8 path: {:?}", path);
            }
        }
    }

    Ok(step_map)
}

// -------------- graph artifacts ---------------

pub struct DataArtifact {
    name: String,
    parent: Option<String>,
    hash: Option<String>,
}

pub enum Node {
    Step,
    DataArtifact,
}

// -------------- new_graph ---------------


// -------------- update_graph ---------------