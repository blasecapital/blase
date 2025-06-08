// /rust/main.rs

mod dag;

fn main() {
    let dir = String::from(r"/workspace/rust/tests/fixtures/steps");
    let step_map = dag::walk_step_dir(&dir);
    println!("The step_map is: {:#?}", step_map);
}
