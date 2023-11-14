use std::collections::HashMap;
use rand::Rng; // rand crate is required in Cargo.toml
use std::process::Command;
use std::io;

pub struct DAGNN {
    a: usize,
    b: usize,
    c: usize,
    ab: usize,
    bc: usize,
    nodes: Vec<f64>,
    expected_output: Vec<f64>,
    weights: (f64, f64, f64, f64),
    links: Vec<Vec<f64>>,
    error: f64,
    fitness: i32,
    current_generation: i32,
    max_generations: i32,
    step_rate: f64,
    unit_tests: Vec<(Vec<i32>, Vec<i32>)>,
    num_tests: Option<usize>,
    previous_links: Option<Vec<Vec<f64>>>, 
    stage: String,
}

impl DAGNN {
    pub fn new(structure: HashMap<&str, usize>) -> Self {
        let a = *structure.get("A").unwrap_or(&0);
        let b = *structure.get("B").unwrap_or(&0);
        let c = *structure.get("C").unwrap_or(&0);
        let ab = a + b;
        let bc = b + c;
        let nodes = vec![0f64; ab + c];
        let expected_output = vec![0f64; c];
        let weights = (0.1, -0.1, -0.1, 0.1);
        let links = (0..ab).map(|i| {
            (0..bc).map(|j| {
                if i < a && j < b {
                    weights.0
                } else if i < a {
                    weights.1
                } else if j + a <= i {
                    0.0
                } else if j < b {
                    weights.2
                } else {
                    weights.3
                }
            }).collect()
        }).collect();

        DAGNN {
            a,
            b,
            c,
            ab,
            bc,
            nodes,
            expected_output,
            weights,
            links,
            error: 0.0,
            fitness: 0,
            current_generation: 0,
            max_generations: 1000000,
            step_rate: 0.001,
            unit_tests: vec![],
            num_tests: None,
            previous_links: None,
            stage: "learning".to_string(),
        }
    }

    pub fn display(&self) {
        Command::new("clear").status().unwrap(); 

        // Display the weights matrix
        for row in &self.links {
            let row_display: String = row.iter().map(|&w| if w != 0.0 { format!("{:.4}  ", w) } else { "         ".to_string() }).collect();
            println!("[{}]", row_display.trim_end_matches(' '));
        }
        println!("\n :: {} ::    {}error: {:.4}    fitness: {} / {}    {}", 
                 self.stage, 
                 if self.stage == "finished" { "".to_string() } else { format!("gen: {}    ", self.current_generation) },
                 self.error, 
                 self.fitness, 
                 self.num_tests.unwrap_or(0),
                 if self.stage == "finished" { format!("size: {}\n", self.links.iter().flatten().filter(|&&w| w != 0.0).count()) } else { "".to_string() });
    }

    pub fn activate(&self, x: f64) -> f64 {
        x * x.tanh()
    }

    pub fn forward(&mut self) {
        for i in 0..self.bc {
            let j = i + self.a;
            self.nodes[j] = self.activate(self.nodes[..j].iter().zip(self.links[..j].iter().map(|row| &row[i])).map(|(x, y)| *x * y).sum());
        }
        self.error += self.nodes[self.ab..].iter().zip(&self.expected_output).map(|(&x, &y)| (x - y).abs() * ((x - y).abs() + 1.0).ln()).sum::<f64>();
        self.fitness += self.nodes[self.ab..].iter().zip(&self.expected_output).map(|(&x, &y)| if (x - 0.5) * (y * 2.0 - 1.0) > 0.0 { 1 } else { 0 }).sum::<i32>();
    }

    pub fn test(&mut self, display: bool) {
        self.error = 0.0;
        self.fitness = 0;
        let unit_tests = self.unit_tests.clone(); // Clone unit_tests to avoid borrowing issue
        for (inputs, outputs) in &unit_tests {
            let input_f64: Vec<f64> = inputs.iter().map(|&val| val as f64).collect(); // Convert i32 to f64
            let output_f64: Vec<f64> = outputs.iter().map(|&val| val as f64).collect(); // Convert i32 to f64
            self.nodes[..self.a].clone_from_slice(&input_f64);
            self.expected_output.clone_from_slice(&output_f64);
            self.forward();
        }
        if self.fitness == self.num_tests.unwrap() as i32 {
            self.previous_links = Some(self.links.clone());
        }
        if display {
            self.display();
        }
    }

    pub fn update_weight(&mut self) {
        self.current_generation += 1;
        let mut rng = rand::thread_rng();
        let i = rng.gen_range(0..self.ab);
        let j = rng.gen_range((i + 1).saturating_sub(self.a)..self.bc);
        if self.links[i][j] != 0.0 {
            // Save the original weight before any tests
            let original_weight = self.links[i][j];

            // Test with a slightly increased weight to estimate gradient
            self.test(false);
            let current_error = self.error;
            self.links[i][j] += self.step_rate;
            self.test(false);
            let new_error = self.error;
            let error_gradient = (current_error - new_error) / self.step_rate;
            // Revert to the original weight
            self.links[i][j] = original_weight;
            self.links[i][j] += error_gradient * self.step_rate / 1.5;
           // println!("Weight updated at ({}, {}): New weight: {}", i, j, self.links[i][j]);
        }
        if self.current_generation % 100 == 0 {
            self.display();
        }
    }

    pub fn learn(&mut self) {
        while self.current_generation < self.max_generations {
            self.update_weight();
    
            // if self.current_generation >= self.max_generations {
            //     println!("\n :: max generations reached. <enter> to continue {} :: ", self.stage);
            //     let mut input_text = String::new();
            //     io::stdin().read_line(&mut input_text).unwrap();
            //     if input_text.trim().is_empty() {
            //         self.current_generation = 0; // Allow continuation from user
            //     }
            // }
    
            if self.fitness == self.num_tests.unwrap().try_into().unwrap() {
                if self.stage == "learning" {
                    self.prune();
                } else if self.stage == "minimizing" {
                    self.stage = "finished".to_string();
                    break; // Exit the learning loop once finished
                }
            }
        }
    
        if self.stage == "learning" {
            println!("\n :: stopped learning ::\n");
        }
    }

    pub fn prune(&mut self) {
        self.stage = "pruning".to_string();
        let mut pruned = false; // Flag to check if any weight was pruned
        while self.stage == "pruning" {
            let mut min_weight = f64::INFINITY;
            let mut min_indices = (0, 0);

            // Find the minimum weight and its indices
            for (i, row) in self.links.iter().enumerate() {
                for (j, &weight) in row.iter().enumerate() {
                    if weight != 0.0 && weight.abs() < min_weight {
                        min_weight = weight.abs();
                        min_indices = (i, j);
                    }
                }
            }

            // Prune the minimum weight if it is not already zero
            if min_weight != f64::INFINITY {
                self.links[min_indices.0][min_indices.1] = 0.0;
                pruned = true;
            }

            // If no weight was pruned, exit the pruning stage
            if !pruned {
                self.stage = "learning".to_string(); // Change stage back to learning as no more pruning possible
                break;
            }

            // Re-evaluate after pruning
            self.learn();
        }

        // Display current state after pruning
        self.test(true);
    }
}

fn main() {
    let mut structure = HashMap::new();
    structure.insert("A", 7);
    structure.insert("B", 3);
    structure.insert("C", 1);

    let mut nn = DAGNN::new(structure);
    nn.unit_tests = (64..128).map(|n| {
        let binary = format!("{:b}", n);
        let input = binary.chars().map(|c| c.to_digit(10).unwrap() as i32).collect::<Vec<_>>();
        let output = binary[1..5].chars().map(|c| c.to_digit(10).unwrap() as i32).collect::<Vec<_>>();
        (input, vec![output[n as usize % 4]])
    }).collect();
    
    nn.num_tests = Some(nn.unit_tests.len());

    // print the unit tests
    // for (inputs, outputs) in &nn.unit_tests {
    //     println!("{:?} -> {:?}", inputs, outputs);
    // }
    nn.learn();
}