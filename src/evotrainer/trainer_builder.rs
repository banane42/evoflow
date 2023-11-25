use std::{error::Error, fmt::Display};
use crate::evonet::EvoNet;
use super::{evotrainer::EvoTrainer, crossover::Strategies};

pub struct TrainerBuilder<'a> {
    parent_strategies: Vec<Strategies>,
    population_size: Option<usize>,
    survival_rate: Option<f64>,
    crossover_rate: Option<f64>,
    mutation_rate: Option<f64>,
    architecture: Option<&'a [usize]>, 
    fitness_function: Option<fn(&mut EvoNet) -> f64>,
}

impl <'a> TrainerBuilder<'a> {

    pub fn new() -> Self {
        Self { 
            parent_strategies: Vec::new(), 
            population_size: None,
            survival_rate: None,
            crossover_rate: None,
            architecture: None,
            mutation_rate: None,
            fitness_function: None
        }
    }

    pub fn build(&self) -> Result<EvoTrainer, TrainerBuildError> {
        let pop_size = self.population_size.ok_or(TrainerBuildError::VariableNotSet(String::from("population_size not set")))?;
        let arch = self.architecture.ok_or(TrainerBuildError::VariableNotSet(String::from("architecture not set")))?;
        let surv_rate = self.survival_rate.unwrap_or(0.0);
        let mut cross_rate = self.crossover_rate.unwrap_or(0.0);
        let mut_rate = self.mutation_rate.unwrap_or(0.0);
        let ft_fn = self.fitness_function.ok_or(TrainerBuildError::VariableNotSet(String::from("fitness_function not set")))?;

        if pop_size <= 1 {
            return Err(TrainerBuildError::ValidationError(String::from("population_size must be greater than 1")));
        }

        for i in arch.iter() {
            if i.eq(&0) {
                return Err(TrainerBuildError::ValidationError(String::from("architecure cannot contain 0's")));
            }
        }

        if surv_rate < 0.0 || surv_rate > 1.0 {
            return Err(TrainerBuildError::ValidationError(String::from("survival_rate must be between 0.0..=1.0")));
        }

        if cross_rate < 0.0 || cross_rate > 1.0 {
            return Err(TrainerBuildError::ValidationError(String::from("crossover_rate must be between 0.0..=1.0")));
        }

        if mut_rate < 0.0 || mut_rate > 1.0 {
            return Err(TrainerBuildError::ValidationError(String::from("mutation_rate must be between 0.0..=1.0")));
        }

        if self.parent_strategies.len() == 0 {
            cross_rate = 0.0;
        }        

        Ok(EvoTrainer::initialize(
            pop_size,
            arch,
            ft_fn,
            surv_rate,
            cross_rate,
            mut_rate,
            self.parent_strategies.clone()
        ))
    }

    pub fn set_population_size(&mut self, size: usize) {
        self.population_size = Some(size);
    }

    pub fn set_architecture(&mut self, architecture: &'a[usize]) {
        self.architecture = Some(architecture);
    }

    pub fn set_survival_rate(&mut self, rate: f64) {
        self.survival_rate = Some(rate);
    }

    pub fn add_parent_selection_strategy(&mut self, strategy: Strategies) {
        match self.parent_strategies.iter().position(|r| r == &strategy) {
            Some(i) => self.parent_strategies[i] = strategy,
            None => self.parent_strategies.push(strategy)
        }
    }

    pub fn set_crossover_rate(&mut self, rate: f64) {
        self.crossover_rate = Some(rate);
    }

    pub fn set_mutation_rate(&mut self, rate: f64) {
        self.mutation_rate = Some(rate);
    }

    pub fn set_fitness_function(&mut self, fit_fn: fn(&mut EvoNet) -> f64) {
        self.fitness_function = Some(fit_fn);
    }

}

#[derive(Debug)]
pub enum TrainerBuildError {
    ValidationError(String),
    VariableNotSet(String),
}

impl Error for TrainerBuildError {}
impl Display for TrainerBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainerBuildError::VariableNotSet(string) => write!(f, "Variable not set. {}", string),
            TrainerBuildError::ValidationError(string) => write!(f, "Valiation error. {}", string)
        }
    }
}