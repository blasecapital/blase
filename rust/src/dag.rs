// /rust/dag.rs

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

pub fn walk_dag(
    db_path: &str,
    start_node: &str,
    node_type: &str
) -> Result<Vec<String>, Box<dyn Error>> {
    let mut dag: Vec<String> = Vec::new();
    // println!("[DEBUG] The dag vec is: {:?}", dag);

    let conn: Connection = Connection::open(db_path)?;
    let node: Node = read_node(&conn, &start_node, &node_type)?;
    // println!("[DEBUG] The node is: {:?}", node);

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
        let mut parent_dag = walk_dag(db_path, &p_hash, &p_type)?;
        dag.extend(parent_dag);
    }
    // println!("[DEBUG] The dag after adding parent_dag is: {:?}", dag);

    Ok(dag)
}