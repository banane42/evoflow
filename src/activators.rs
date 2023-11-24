#![allow(dead_code)]

#[derive(Clone, Copy)]
pub enum Type {
    Sigmoid,
    Tanh,
    Relu,
    Custom
}

#[derive(Clone, Copy)]
pub struct ActivationContainer {
    pub func: fn(f64) -> f64
}

pub fn sigm(x: f64) -> f64{ 1.0/(1.0 + x.exp()) }

pub fn tanh(x: f64) -> f64{
    x.tanh()
}

pub fn relu(x: f64) -> f64{
    f64::max(0.0, x)
}