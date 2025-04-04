use crate::model::Model;

pub struct Task {
    pub data: Vec<(f64, f64)>,
    pub model: Model,
}

pub struct Gradient {
    pub grad_a: f64,
    pub grad_b: f64,
}
