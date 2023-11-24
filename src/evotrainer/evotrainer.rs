use std::collections::BinaryHeap;
use crate::evonet::EvoNet;

pub struct EvoTrainer<'a> {
    population: Vec<EvoNet>,
    params: TrainerParams<'a>
}

pub struct TrainerParams<'a> {
    size: usize, 
    architecture: &'a[i32], 
    fitness_fn: fn(&mut EvoNet) -> f64,
    survival_percentile: f64,
    mutation_frequency: f64,
    crossover_rate: f64,
    prime_parent_rate: f64,
}

impl <'a> TrainerParams <'a> {
    pub fn build(
        population_size: usize,
        architecture: &'a[i32],
        fitness_function: fn(&mut EvoNet) -> f64,
        survival_percentile: f64,
        mutation_frequency: f64,
        crossover_rate: f64,
        prime_parent_rate: f64
    ) -> Result<Self, &str> {

        if population_size <= 1 {
            return Err("population_size must be greater than 1");
        }

        if survival_percentile < 0.0 || survival_percentile > 1.0 {
            return Err("survival_percentile out of bounds (0.0..=1.0)");
        }

        if mutation_frequency < 0.0 || mutation_frequency > 1.0 {
            return Err("mutation_frequency out of bounds (0.0..=1.0)");
        }

        if crossover_rate < 0.0 || crossover_rate > 1.0 {
            return Err("crossover_rate out of bounds (0.0..=1.0)");
        }

        if prime_parent_rate < 0.0 || prime_parent_rate > 1.0 {
            return Err("prime_parent_rate out of bounds (0.0..=1.0)");
        }

        Ok(Self { 
            size: population_size,
            architecture, 
            fitness_fn: fitness_function,
            survival_percentile,
            mutation_frequency,
            crossover_rate,
            prime_parent_rate
        })
    }
}

#[derive(Debug)]
pub struct FitnessPair {
    pub fitness: f64,
    pub index: usize
}

impl HasFitness for FitnessPair {
    fn get_fitness(&self) -> f64 {
        self.fitness
    }
}

impl Ord for FitnessPair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.fitness.total_cmp(&other.fitness)
    }
}

impl PartialOrd for FitnessPair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for FitnessPair {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Eq for FitnessPair {}

impl <'a> EvoTrainer<'a> {

    pub fn initialize(t_params: TrainerParams<'a>) -> Self {
        let mut pop_vec = Vec::with_capacity(t_params.size);
        Self::spawn_population(&mut pop_vec, t_params.architecture, t_params.fitness_fn);
        Self { 
            population: pop_vec,
            params: t_params
        }
    }

    pub fn show_population(&self) {
        self.population.iter().for_each(|net| {
            println!("{}", net);
        });
    }

    pub fn show_individual(&self, index: usize) -> Result<(), ()> {
        match self.population.get(index) {
            Some(net) => {
                println!("{}", net); 
                Ok(())
            },
            None => Err(()),
        }
    }

    pub fn extract_best(&self) -> EvoNet {
        let mut ex_net: &EvoNet = self.population.first().unwrap();
        self.population.iter().for_each(|net| {
            if net.get_fitness() > ex_net.get_fitness() {
                ex_net = net;
            }
        });

        ex_net.clone()
    }

    pub fn train(&mut self, generations: usize) {
        (0..generations).for_each(|_| {
            let mut pop_fitness = self.calculate_pop_fitness();
            self.create_next_gen(&mut pop_fitness, self.params.survival_percentile);
            self.mutate_population();
        });
    }

    pub fn calculate_pop_fitness(&mut self) -> Vec<FitnessPair> {
        let mut fitnesses: BinaryHeap<FitnessPair> = BinaryHeap::new();
        for (i, net) in self.population.iter_mut().enumerate() {
            let ft_score = (self.params.fitness_fn)(net);
            net.set_fitness(ft_score);
            fitnesses.push(FitnessPair { fitness: ft_score, index: i })
        }
        //Sorted from low fitness to high fitness
        return fitnesses.into_sorted_vec();
    }

    fn create_next_gen(&mut self, fitness_pairs: &mut Vec<FitnessPair>, survival_rate: f64) {
        let dead_pop: Vec<_> = fitness_pairs.drain(0..(fitness_pairs.len() as f64 * (1.0 - survival_rate)) as usize).collect();
        let (crossover_pop, copy_pop) = dead_pop.split_at((dead_pop.len() as f64 * self.params.crossover_rate) as usize);

        // self.generate_from_rank_crossover(fitness_pairs, crossover_pop);
        // self.generate_from_tournament_crossover(fitness_pairs, crossover_pop);
        self.generate_from_copy(fitness_pairs, copy_pop);
    }

    // fn generate_from_rank_crossover(&mut self, fitness_pairs: &Vec<FitnessPair>, crossover_pop: &[FitnessPair]) {
    //     let mut rng = rand::thread_rng();
    //     let prime_parent_count = (fitness_pairs.len() as f64 * self.params.prime_parent_rate).max(1.0) as usize;

    //     crossover_pop.iter().for_each(|pair| {
    //         let p_a_i = rng.gen_range((fitness_pairs.len() - prime_parent_count)..fitness_pairs.len());
    //         let mut p_b_i = rng.gen_range((fitness_pairs.len() - prime_parent_count)..fitness_pairs.len());
    //         while p_a_i == p_b_i {
    //             p_b_i = rng.gen_range((fitness_pairs.len() - prime_parent_count)..fitness_pairs.len());
    //         }
            
    //         let p_a = fitness_pairs.get(p_a_i).unwrap();
    //         let p_b = fitness_pairs.get(p_b_i).unwrap();
            
    //         self.population[pair.index] = self.crossover(
    //             p_a.index, 
    //             p_b.index,
    //              p_a.fitness, 
    //              p_b.fitness
    //         );
    //     });
    // }

    fn generate_from_copy(&mut self, fitness_pairs: &Vec<FitnessPair>, copy_pop: &[FitnessPair]) {
        let prime_parent_count = (fitness_pairs.len() as f64 * self.params.prime_parent_rate).max(1.0) as usize;
        let mut i: usize = 1;
        
        // let pop_deviation = Self::calc_std_deviation(&fitness_pairs);
        // let ratio = pop_deviation / self.params.std_deviation;
        // let mut_variance = 1.0 - 1.0_f64.min(ratio);
        let mut_variance = 0.0;

        copy_pop.iter().for_each(|pair| {
            let parent = &fitness_pairs[fitness_pairs.len() - i];
            i += 1;
            if i >= prime_parent_count {
                i = 1;
            }
            let mut child = self.population[parent.index].clone();
            child.mutate(self.params.mutation_frequency + mut_variance);
            self.population[pair.index] = child;
        });
    }

    // fn generate_from_tournament_crossover(&mut self, fitness_pairs: &Vec<FitnessPair>, crossover_pop: &[FitnessPair]) {
    //     let mut rng = rand::thread_rng();

    //     crossover_pop.iter().for_each(|pair| {
    //         let mut p_a_i = rng.gen_range(0..fitness_pairs.len());
    //         for _ in 1..self.params.tournament_size {
    //             let challenger = rng.gen_range(0..fitness_pairs.len());
    //             if fitness_pairs[challenger].fitness > fitness_pairs[p_a_i].fitness {
    //                 p_a_i = challenger;
    //             }
    //         }

    //         let mut p_b_i = rng.gen_range(0..fitness_pairs.len());
    //         for _ in 1..self.params.tournament_size {
    //             let challenger = rng.gen_range(0..fitness_pairs.len());
    //             if fitness_pairs[challenger].fitness > fitness_pairs[p_b_i].fitness {
    //                 p_b_i = challenger;
    //             }
    //         }

    //         let p_a = fitness_pairs.get(p_a_i).unwrap();
    //         let p_b = fitness_pairs.get(p_b_i).unwrap();

    //         self.population[pair.index] = self.crossover(
    //             p_a.index, 
    //             p_b.index, 
    //             p_a.fitness, 
    //             p_b.fitness
    //         );
    //     });
    // }

    // fn create_generation_from_roulette(&mut self, fitness_pairs: &mut Vec<FitnessPair>) {
        
    // }

    fn spawn_population(pop_vec: &mut Vec<EvoNet>, architecture: &[i32], fitness_fn: fn(&mut EvoNet) -> f64) {
        (0..pop_vec.capacity()).for_each(|_| {
            let mut spawn = EvoNet::new(architecture);
            let spawn_fit = (fitness_fn)(&mut spawn);
            spawn.set_fitness(spawn_fit);
            pop_vec.push(
                spawn
            );
        })
    }

    fn crossover(&self, parent_a_idx: usize, parent_b_idx: usize, p1_fitness: f64, p2_fitness: f64) -> EvoNet {
        let p_a = self.population.get(parent_a_idx).unwrap();
        let p_b = self.population.get(parent_b_idx).unwrap();
        let mut child = EvoNet::from_parents(p_a, p_b, p1_fitness, p2_fitness);
        let c_fit = (self.params.fitness_fn)(&mut child);
        child.set_fitness(c_fit);
        child
    }

    fn mutate_population(&mut self) {
        // let pop_deviation = Self::calc_std_deviation(&self.population);
        // let ratio = pop_deviation / self.params.std_deviation;
        // let ratio = 1.0;
        // let mut_variance = 1.0 - 1.0_f64.min(ratio);

        self.population.iter_mut().for_each(|net| 
            net.mutate(self.params.mutation_frequency)
        )
    }

    fn calc_std_deviation<T: HasFitness>(data: &Vec<T>) -> f64 {
        let n = data.len() as f64;
    
        let (sum, sum_sq) = data.iter().fold((0.0, 0.0), |(sum, sum_sq), pair| {
            (sum + pair.get_fitness(), sum_sq + pair.get_fitness().powf(2.0))
        });

        let mean = sum / n;
        ((sum_sq / n) - (mean * mean)).sqrt()
    }
}

pub trait HasFitness {
    fn get_fitness(&self) -> f64;
}