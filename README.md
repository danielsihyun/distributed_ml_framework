Overview

This project simulates a distributed machine learning training framework for a simple linear regression model. A synthetic dataset is created based on the equation y = 2.0x + 1.0 with added noise. The framework splits the dataset into chunks and uses multiple worker threads to compute gradients concurrently. A scheduler collects the gradients and updates the model parameters using gradient descent.

Project Structure

• Cargo.toml - Contains project configuration and dependency management (includes the rand crate). 
• src/main.rs - The entry point; it generates synthetic data, sets training parameters, and initiates the training loop. 
• src/model.rs - Defines the linear regression model with methods for prediction, loss computation, and parameter updates. 
• src/scheduler.rs - Manages the training loop, distributes tasks to worker threads, collects gradients, and updates the model. 
• src/worker.rs - Contains the logic for computing gradients on data chunks. 
• src/communication.rs - Defines data structures (tasks and gradients) used for communication between the scheduler and workers.

Installation and Running

Ensure that Rust is installed on your system.
Open a terminal and change directory to the project folder.
Run "cargo run" to compile and execute the project.
The program will display the loss value for each training iteration and the final model parameters after training completes.

Customization

You may adjust parameters such as the number of worker threads, number of iterations, and learning rate in the main function. The synthetic dataset can also be modified to experiment with different noise levels or true model parameters.

Dependencies

The project uses the following dependency: • rand (for generating synthetic data with noise).
