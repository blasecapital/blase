// /rust/dag.rs

//! DAG Utilities for Hash-Based Provenance Tracking
//!
//! This module defines the data structures and functions needed to represent and
//! traverse a directed acyclic graph (DAG) of computational steps and data artifacts.
//!
//! It is primarily used to store and query provenance data in a SQLite database.
//!
//! # Components
//! - `Step` and `DataArtifact`: Represent the two types of nodes in the DAG.
//! - `Node`: Enum wrapper around either a `Step` or `DataArtifact`.
//! - `read_node`: Reads a node from the database by hash and type.
//! - `walk_dag`: Traverses the DAG recursively from a given starting node.

use std::error::Error;

use rusqlite::{params, Connection, Result};
use serde_json::Value;
use serde::{Deserialize, Serialize};
use serde::de::Error as SerdeError;
use serde::de::DeserializeOwned;

#[derive(Deserialize, Debug)]
pub struct Step {
    step_hash: String,
    step_id: String,
    function: String,
    parent: Option<String>,
    parent_type: Option<String>,
    params: String,
    timestamp_start: String,
    timestamp_end: String,
    status: String,
    outputs: String,
}

#[derive(Deserialize, Debug)]
pub struct DataArtifact {
    data_hash: String,
    source_path: String,
    parent: Option<String>,
    parent_type: Option<String>,
    logged_by: String,
    metadata: String,
}

#[derive(Debug, serde::Deserialize)]
pub enum Node {
    Step(Step),
    DataArtifact(DataArtifact),
}

impl Node {
    pub fn parse_params<T: DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        if let Node::Step(s) = self {
            serde_json::from_str(&s.params)
        } else {
            Err(serde_json::Error::custom("Node is not a Step"))
        }
    }

    pub fn parse_outputs<T: DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        if let Node::Step(s) = self {
            serde_json::from_str(&s.outputs)
        } else {
            Err(serde_json::Error::custom("Node is not a Step"))
        }
    }

    pub fn parse_metadata<T: DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        if let Node::DataArtifact(d) = self {
            serde_json::from_str(&d.metadata)
        } else {
            Err(serde_json::Error::custom("Node is not a DataArtifact"))
        }
    }
}

/// Reads a single node (either `Step` or `DataArtifact`) from the SQLite database by its hash.
///
/// # Parameters
/// - `conn`: An active `rusqlite::Connection`.
/// - `node_hash`: Hash of the node to retrieve.
/// - `node_type`: Either `"step"` or `"data"`.
///
/// # Returns
/// A `Node` enum containing the corresponding `Step` or `DataArtifact`.
///
/// # Errors
/// Returns an error if the node cannot be found, the type is unknown, or the database read fails.
pub fn read_node(
    conn: &Connection,
    node_hash: &str,
    node_type: &str
) -> Result<Node, Box<dyn Error>> {
    // Create dynamic query inputs
    let mut table = String::new();
    let mut hash_col = String::new();

    if node_type == "step" {
        table.push_str("steps");
        hash_col.push_str("step_hash");
    } else {
        table.push_str("data");
        hash_col.push_str("data_hash");
    }

    // Query the node data
    let query = format!(
        "SELECT * FROM {} WHERE {} = ?1",
        table, hash_col
    );
    let mut stmt = conn.prepare(&query)?;
    let node = match node_type {
        "step" => {
            let step = stmt.query_row([node_hash], |row| {
                Ok(Step {
                    step_hash: row.get(0)?,
                    step_id: row.get(1)?,
                    function: row.get(2)?,
                    parent: row.get(3)?,
                    parent_type: row.get(4)?,
                    params: row.get(5)?,
                    timestamp_start: row.get(6)?,
                    timestamp_end: row.get(7)?,
                    status: row.get(8)?,
                    outputs: row.get(9)?,
                })
            })?;
            Node::Step(step)
        }
        "data" => {
            let data = stmt.query_row([node_hash], |row| {
                Ok(DataArtifact {
                    data_hash: row.get(0)?,
                    source_path: row.get(1)?,
                    parent: row.get(2)?,
                    parent_type: row.get(3)?,
                    logged_by: row.get(4)?,
                    metadata: row.get(5)?,
                })
            })?;
            Node::DataArtifact(data)
        }
        _ => return Err(format!("Unknown node type: {}", node_type).into()),
    };
    Ok(node)
}

/// Recursively walks the DAG from a given node, collecting identifiers.
///
/// Starts at the given node and follows parent references until the root is reached.
///
/// # Parameters
/// - `db_path`: Path to the SQLite database.
/// - `start_node`: Hash of the starting node.
/// - `node_type`: Type of the starting node (`"step"` or `"data"`).
///
/// # Returns
/// A vector of strings containing either step IDs or data source paths, from the
/// start node up to the root.
///
/// # Errors
/// Returns an error if any database operations fail or if the DAG traversal encounters unknown node types.
///
/// # Example
/// 
/// let lineage = walk_dag("mydb.sqlite", "abc123", "step").unwrap();
/// for node in lineage {
///     println!("{}", node);
/// }
/// 
pub fn walk_dag(
    db_path: &str,
    start_node: &str,
    node_type: &str
) -> Result<Vec<String>, Box<dyn Error>> {
    let conn: Connection = Connection::open(db_path)?;
    walk_dag_inner(&conn, start_node, node_type)
}

fn walk_dag_inner(conn: &Connection, start_node: &str, node_type: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let mut dag: Vec<String> = Vec::new();

    let node: Node = read_node(&conn, &start_node, &node_type)?;

    let parent_hash: Option<String> = match &node {
        Node::Step(step) => step.parent.clone(),
        Node::DataArtifact(data) => data.parent.clone(),
    };
    let parent_type: Option<String> = match &node {
        Node::Step(step) => step.parent_type.clone(),
        Node::DataArtifact(data) => data.parent_type.clone(),
    };

    match &node {
        Node::Step(step) => dag.push(step.step_id.clone()),
        Node::DataArtifact(data) => dag.push(data.source_path.clone()),
    }

    if let (Some(p_hash), Some(p_type)) = (parent_hash, parent_type) {
        let mut parent_dag = walk_dag_inner(conn, &p_hash, &p_type)?;
        dag.extend(parent_dag);
    }

    Ok(dag)
}