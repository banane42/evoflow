use rand::{self, Rng};

use crate::evotrainer::evotrainer::FitnessPair;

pub trait ParentSelectionStrategy {
    /// Weight representing how much this strategy should be used
    /// in relation to other strategies being employed by the trainer
    fn get_weight(&self) -> usize;

    /// Takes the available parents and the population to be replaced
    /// by the offspring and returns the parents that will replace that 
    /// member in the population
    fn create_offspring(&self, parent_fitness_pairs: &Vec<FitnessPair>, crossover_pop: &[FitnessPair]) -> Vec<CrossoverFamily>;
}

pub struct CrossoverFamily {
    child_index: usize,
    parent_a_index: usize,
    parent_b_index: usize,
    parent_a_fitness: f64,
    parent_b_fitness: f64
}

/// Randomly selects parents from the set amount of rounds
/// Picking the best out of two tournements to be parents 
pub struct TournamentSelection {
    weight: usize,
    rounds: usize
}

impl ParentSelectionStrategy for TournamentSelection {
    fn get_weight(&self) -> usize {
        self.weight
    }

    fn create_offspring(&self, parent_fitness_pairs: &Vec<FitnessPair>, crossover_pop: &[FitnessPair]) -> Vec<CrossoverFamily> {
        let mut rng = rand::thread_rng();
        let mut children: Vec<CrossoverFamily> = Vec::with_capacity(crossover_pop.len());

        crossover_pop.iter().for_each(|pair| {
            let mut p_a_i = rng.gen_range(0..parent_fitness_pairs.len());
            for _ in 1..self.rounds {
                let challenger = rng.gen_range(0..parent_fitness_pairs.len());
                if parent_fitness_pairs[challenger].fitness > parent_fitness_pairs[p_a_i].fitness {
                    p_a_i = challenger;
                }
            }

            let mut p_b_i = rng.gen_range(0..parent_fitness_pairs.len());
            for _ in 1..self.rounds {
                let challenger = rng.gen_range(0..parent_fitness_pairs.len());
                if parent_fitness_pairs[challenger].fitness > parent_fitness_pairs[p_b_i].fitness {
                    p_b_i = challenger;
                }
            }

            let p_a = &parent_fitness_pairs[p_a_i];
            let p_b = &parent_fitness_pairs[p_b_i];

            children.push(CrossoverFamily { 
                child_index: pair.index,
                parent_a_index: p_a.index,
                parent_b_index: p_b.index,
                parent_a_fitness: p_a.fitness, 
                parent_b_fitness: p_b.fitness,
            });
        });

        children
    }
}

pub struct PrimeParentStrategy {
    weight: usize,
    rate: f64
}

/// Randomly selects from the top percent of parents 
/// Top percent is defined by the rate variable
impl ParentSelectionStrategy for PrimeParentStrategy {
    fn get_weight(&self) -> usize {
        self.weight
    }

    fn create_offspring(&self, parent_fitness_pairs: &Vec<FitnessPair>, crossover_pop: &[FitnessPair]) -> Vec<CrossoverFamily> {
        let mut rng = rand::thread_rng();
        let mut children: Vec<CrossoverFamily> = Vec::with_capacity(crossover_pop.len());
        let prime_parent_count = (parent_fitness_pairs.len() as f64 * self.rate).max(1.0) as usize;

        crossover_pop.iter().for_each(|pair| {
            let p_a_i = rng.gen_range((parent_fitness_pairs.len() - prime_parent_count)..parent_fitness_pairs.len());
            let mut p_b_i = rng.gen_range((parent_fitness_pairs.len() - prime_parent_count)..parent_fitness_pairs.len());
            while p_a_i == p_b_i {
                p_b_i = rng.gen_range((parent_fitness_pairs.len() - prime_parent_count)..parent_fitness_pairs.len());
            }
            
            let p_a = &parent_fitness_pairs[p_a_i];
            let p_b = &parent_fitness_pairs[p_b_i];
            
            children.push(CrossoverFamily { 
                child_index: pair.index,
                parent_a_index: p_a.index,
                parent_b_index: p_b.index,
                parent_a_fitness: p_a.fitness, 
                parent_b_fitness: p_b.fitness,
            });
        });

        children
    }
}

/// Randomly selects parents weighted by the fitness ratio of
/// the parent compared to all other parents
pub struct RouletteStrategy {
    weight: usize
}

impl ParentSelectionStrategy for RouletteStrategy {
    fn get_weight(&self) -> usize {
        self.weight
    }

    fn create_offspring(&self, parent_fitness_pairs: &Vec<FitnessPair>, crossover_pop: &[FitnessPair]) -> Vec<CrossoverFamily> {
        let fitness_sum = parent_fitness_pairs.iter().fold(0.0, |sum, pair| sum + pair.fitness);
        let mut rng = rand::thread_rng();
        let mut children: Vec<CrossoverFamily> = Vec::with_capacity(crossover_pop.len());

        crossover_pop.iter().for_each(|pair| {
            let p_a_i = (rng.gen_range(0.0..fitness_sum) * parent_fitness_pairs.len() as f64) as usize;
            let p_b_i = (rng.gen_range(0.0..fitness_sum) * parent_fitness_pairs.len() as f64) as usize;

            let p_a = &parent_fitness_pairs[p_a_i];
            let p_b = &parent_fitness_pairs[p_b_i];

            children.push(CrossoverFamily { 
                child_index: pair.index,
                parent_a_index: p_a.index,
                parent_b_index: p_b.index,
                parent_a_fitness: p_a.fitness, 
                parent_b_fitness: p_b.fitness,
            })
        });

        children
    }
}