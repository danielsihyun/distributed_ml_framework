use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use crate::model::Model;
use crate::communication::{Task, Gradient};
use crate::worker::compute_gradient;

pub fn run_training(data: Vec<(f64, f64)>, num_workers: usize, iterations: usize, learning_rate: f64) -> Model {
    // create channels for tasks/gradients
    let (task_tx, task_rx) = mpsc::channel::<Task>();
    let (grad_tx, grad_rx) = mpsc::channel::<Gradient>();

    let task_rx = Arc::new(Mutex::new(task_rx));

    // spawn worker threads
    let mut handles = Vec::new();
    for _ in 0..num_workers {
        let task_rx = Arc::clone(&task_rx);
        let grad_tx = grad_tx.clone();
        let handle = thread::spawn(move || {
            loop {
                let task_option = {
                    let lock = task_rx.lock().unwrap();
                    lock.recv()
                };
                match task_option {
                    Ok(task) => {
                        let grad = compute_gradient(task);
                        if grad_tx.send(grad).is_err() {
                            break;
                        }
                    },
                    Err(_) => break,
                }
            }
        });
        handles.push(handle);
    }

    drop(grad_tx);

    let mut model = Model::new();

    // split data into chunks for workers
    let chunk_size = (data.len() as f64 / num_workers as f64).ceil() as usize;
    let data_chunks: Vec<Vec<(f64, f64)>> = data.chunks(chunk_size)
        .map(|chunk| chunk.to_vec())
        .collect();

    for iter in 0..iterations {
        println!("Iteration {}: Current Loss = {}", iter, model.loss(&data));
        // assign tasks to each worker w current model and data chunk
        for i in 0..num_workers {
            let chunk = data_chunks.get(i % data_chunks.len()).unwrap().clone();
            let task = Task {
                data: chunk,
                model,
            };
            if let Err(e) = task_tx.send(task) {
                eprintln!("Error sending task: {}", e);
            }
        }

        // collect gradients from all workers
        let mut total_grad_a = 0.0;
        let mut total_grad_b = 0.0;
        for _ in 0..num_workers {
            if let Ok(grad) = grad_rx.recv() {
                total_grad_a += grad.grad_a;
                total_grad_b += grad.grad_b;
            }
        }
        // avg gradients and update model
        total_grad_a /= num_workers as f64;
        total_grad_b /= num_workers as f64;
        model.update(total_grad_a, total_grad_b, learning_rate);
    }

    drop(task_tx);
    for handle in handles {
        handle.join().unwrap();
    }

    model
}
