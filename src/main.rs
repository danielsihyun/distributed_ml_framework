mod scheduler;
mod worker;
mod model;
mod communication;

use scheduler::run_training;
use rand::random;

fn main() {
    println!("Distributed ML Training Framework Starting...");

    // making synthetic data
    // y = 2x + 1 with random noise
    let mut data = Vec::new();
    for i in 0..100 {
        let x = i as f64;
        let noise = (random::<f64>() - 0.5) * 5.0;
        let y = 2.0 * x + 1.0 + noise;
        data.push((x, y));
    }

    // training parameters
    let num_workers = 4;
    let iterations = 20;
    let learning_rate = 0.0001;

    let final_model = run_training(data, num_workers, iterations, learning_rate);

    println!("Final Model: a = {}, b = {}", final_model.a, final_model.b);
}