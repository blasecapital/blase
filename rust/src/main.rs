// /rust/main.rs

mod dag;

fn main() {
    let db_path = String::from(r"/workspace/rust/tests/fixtures/nodes.db");
    let start_step = String::from("d74360136646b829b92dc2b6ed117fcdae2744c6076e7d551a698ccc2093a82f");
    let node_type = String::from("step");
    let outcome = dag::walk_dag(&db_path, &start_step, &node_type);
    println!("The outcome is: {:?}", outcome);
}
