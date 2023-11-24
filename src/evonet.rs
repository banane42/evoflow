use std::fmt::Display;
use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

use crate::{activators::{self, ActivationContainer}, evotrainer::evotrainer::HasFitness};

#[derive(Clone)]
struct Layer {
    v: Vec<f64>,
    y: Vec<f64>,
    w: Vec<Vec<f64>>,
}

impl Layer {
    fn new(amount: i32, input: i32) -> Layer {
        let mut nl = Layer {v: vec![], y: vec![], w: Vec::new()};
        let mut v: Vec<f64>;
        for _ in 0..amount {
            nl.y.push(0.0);
            nl.v.push(0.0);

            v = Vec::new();
            for _ in 0..input + 1 {
                v.push(2f64 * rand::random::<f64>() - 1f64);
            }

            nl.w.push(v);
        }
        return nl;
    }

    fn new_from_parents(l1: &Layer, l2: &Layer, p1_fitness: f64, p2_fitness: f64) -> Layer {
        let mut nl = Layer {
            v: l1.v.clone(),
            y: l1.y.clone(),
            w: l1.w.clone(),
        };

        let mut rng = thread_rng();
        let fitness_ratio = p1_fitness / (p1_fitness + p2_fitness);

        for i in 0..nl.w.len() {
            for j in 0..nl.w[i].len() {
                if rng.gen_range(0.0..=1.0) > fitness_ratio {
                    nl.w[i][j] = l2.w[i][j];
                }
            }
        }

        nl
    }
}

impl Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut disp_str = String::from("");
        for i in 0..self.w.len() {
            disp_str.push('[');
            for j in 0..self.w[i].len() {
                disp_str.push_str(&format!("{}, ", &self.w[i][j].to_string()));
            }
            disp_str.push(']');
            disp_str.push('\n');
        }
        write!(f, "{}", disp_str)
    }
}

#[derive(Clone)]
pub struct EvoNet {
    layers: Vec<Layer>,
    act_type: activators::Type,
    act: ActivationContainer,
    fitness: f64
}

impl EvoNet {
    pub fn new(architecture: &[i32]) -> EvoNet {
        let mut nn = EvoNet {
            layers: Vec::new(),
            act: ActivationContainer{ func: activators::tanh },
            act_type: activators::Type::Tanh,
            fitness: 0.0,
        };

        for i in 1..architecture.len() {
            nn.layers.push(Layer::new(architecture[i], architecture[i - 1]))
        }

        return nn;
    }

    pub fn from_parents(p1: &EvoNet, p2: &EvoNet, p1_fitness: f64, p2_fitness: f64) -> EvoNet {
        let mut nn = EvoNet {
            layers: Vec::new(),
            act: p1.act,
            act_type: p1.act_type,
            fitness: 0.0
        };

        for i in 0..p1.layers.len() {
            nn.layers.push(Layer::new_from_parents(
                &p1.layers[i], 
                &p2.layers[i], 
                p1_fitness, p2_fitness
            ))
        }

        nn
    }

    pub fn set_fitness(&mut self, ft: f64) {
        self.fitness = ft;
    }

    fn forward(&mut self, x: &Vec<f64>) {
        let mut sum: f64;

        for j in 0..self.layers.len() {
            if j == 0 {
                for i in 0..self.layers[j].v.len(){
                    sum = 0.0;
                    for k in 0..x.len(){
                        sum += self.layers[j].w[i][k] * x[k];
                    }
                    self.layers[j].v[i] = sum;
                    self.layers[j].y[i] = (self.act.func)(sum);
                }
            } else if j == self.layers.len() - 1 {
                for i in 0..self.layers[j].v.len(){
                    sum = self.layers[j].w[i][0];
                    for k in 0..self.layers[j - 1].y.len(){
                        sum += self.layers[j].w[i][k + 1] * self.layers[j - 1].y[k];
                    }
                    self.layers[j].v[i] = sum;
                    self.layers[j].y[i] = sum;
                }
            } else {
                for i in 0..self.layers[j].v.len(){
                    sum = self.layers[j].w[i][0];
                    for k in 0..self.layers[j - 1].y.len(){
                        sum += self.layers[j].w[i][k + 1] * self.layers[j - 1].y[k];
                    }
                    self.layers[j].v[i] = sum;
                    self.layers[j].y[i] = (self.act.func)(sum);
                }
            }
        }
    }

    #[allow(non_snake_case)]
    pub fn calc(&mut self, X: &[f64]) -> &[f64]{
        let mut x = X.to_vec();

        x.insert(0, 1f64);

        self.forward(&x);
        &self.layers[self.layers.len() - 1].y
    }

    pub fn mutate(&mut self, frequency: f64) {
        let mut rng = thread_rng();
        for layer in 0..self.layers.len() {
            for neuron in 0..self.layers[layer].v.len() {
                for weight in 0..self.layers[layer].w[neuron].len() {
                    if rng.gen_range(0.0..=1.0) <= frequency {
                        let amount = 0.1 * rng.sample::<f64, _>(StandardNormal);
                        self.layers[layer].w[neuron][weight] = (self.layers[layer].w[neuron][weight] + amount).clamp(0.0, 1.0);
                    }
                }
            }
        }
    }
    
}

impl Display for EvoNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut net_str = String::from("");

        self.layers.iter().for_each(|l| {
            net_str.push_str("layer\n");
            net_str.push_str(&l.to_string());
        });

        write!(f, "fitness: {}\n{}", self.fitness, net_str)
    }
}

impl HasFitness for EvoNet {
    fn get_fitness(&self) -> f64 {
        self.fitness
    }
}