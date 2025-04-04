#[derive(Clone, Copy, Debug)]
pub struct Model {
    pub a: f64,
    pub b: f64,
}

impl Model {
    pub fn new() -> Self {
        Self { a: 0.0, b: 0.0 }
    }

    pub fn predict(&self, x: f64) -> f64 {
        self.a * x + self.b
    }

    // mse loss
    pub fn loss(&self, data: &[(f64, f64)]) -> f64 {
        let n = data.len() as f64;
        let sum: f64 = data.iter()
            .map(|(x, y)| {
                let diff = self.predict(*x) - *y;
                diff * diff
            })
            .sum();
        sum / n
    }

    // update parameters
    pub fn update(&mut self, grad_a: f64, grad_b: f64, learning_rate: f64) {
        self.a -= learning_rate * grad_a;
        self.b -= learning_rate * grad_b;
    }
}
