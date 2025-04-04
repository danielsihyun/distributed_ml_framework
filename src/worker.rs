use crate::communication::{Task, Gradient};

pub fn compute_gradient(task: Task) -> Gradient {
    let data = task.data;
    let model = task.model;
    let n = data.len() as f64;
    if n == 0.0 {
        return Gradient { grad_a: 0.0, grad_b: 0.0 };
    }
    let mut sum_grad_a = 0.0;
    let mut sum_grad_b = 0.0;
    for (x, y) in data {
        let prediction = model.predict(x);
        let error = prediction - y;
        sum_grad_a += error * x;
        sum_grad_b += error;
    }
    Gradient {
        grad_a: 2.0 * sum_grad_a / n,
        grad_b: 2.0 * sum_grad_b / n,
    }
}
