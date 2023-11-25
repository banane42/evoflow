use evoflow::{evotrainer::{trainer_builder::TrainerBuilder, crossover::{PrimeParentStrategy, Strategies}}, evonet::EvoNet};

fn main() {
    // let params = TrainerParams::build(
    //     1000, 
    //     &[2, 2, 1],
    //     xor_fit_fn,
    //     0.60,
    //     0.01,
    //     1.0
    // ).expect("Training params wrong");

    // let mut trainer = TrainerBuilder::new()
    //     .set_population_size(1000)
    //     .build()
    //     .unwrap_or_else(|e| panic!("{}", e));
    let mut builder = TrainerBuilder::new();
    builder.set_architecture(&[2, 2, 1]);
    builder.set_population_size(1000);
    builder.set_fitness_function(xor_fit_fn);
    builder.add_parent_selection_strategy(Strategies::PrimeParent(PrimeParentStrategy {
        weight: 1,
        rate: 0.1,
    }));

    let mut trainer = builder.build().unwrap_or_else(|e| panic!("{}", e));

    loop {
        let mut input = String::new();
        let _ = std::io::stdin().read_line(& mut input).unwrap();

        let parts: Vec<&str> = input.as_str().trim().split(char::is_whitespace).collect();

        match parts[0] {
            "exit" | "e" => {
                println!("Exiting...");
                break;
            }
            "train" | "t" => {
                match parts.get(1) {
                    Some(val) => {
                        match val.parse::<usize>() {
                            Ok(i) => {
                                trainer.train(i);
                            },
                            Err(_) => eprintln!("Could not parse into usize")
                        }
                    },
                    None => eprintln!("command not supplied with number of generations to train")
                }
            }
            "display" | "d" => {
                match parts.get(1) {
                    Some(val) => {
                        match val.parse::<usize>() {
                            Ok(index) => {
                                _ = trainer.show_individual(index);
                            },
                            Err(_) => eprintln!("Could not parse into usize"),
                        }
                    },
                    None => trainer.show_population(),
                }
            }
            "extract" | "ex" => {
                let mut best = trainer.extract_best();
                println!("Extracting Best. Result");
                xor_fit_fn_print(&mut best);
            }
            _ => {
                println!("Unknown Input: {:?}", parts);
            }
        }

    }

}

fn xor_fit_fn(net: &mut EvoNet) -> f64 {   
    let a0 = net.calc(&[0.0, 0.0])[0]; // Should be 0
    let a1 = net.calc(&[0.0, 1.0])[0]; // Should be 1
    let a2 = net.calc(&[1.0, 0.0])[0]; // Should be 1
    let a3 = net.calc(&[1.0, 1.0])[0]; // Should be 0

    let fit0 = if a0.round() == 0.0 {
        1.0
    } else { -1.0 };
    let fit1 = if a1.round() == 1.0 {
        1.0
    } else { -1.0 };
    let fit2 = if a2.round() == 1.0 {
        1.0
    } else { -1.0 };
    let fit3 = if a3.round() == 0.0 {
        1.0
    } else { -1.0 };

    fit0 + fit1 + fit2 + fit3
}

fn xor_fit_fn_print(net: &mut EvoNet){   
    let a0 = net.calc(&[0.0, 0.0])[0]; // Should be 0
    let a1 = net.calc(&[0.0, 1.0])[0]; // Should be 1
    let a2 = net.calc(&[1.0, 0.0])[0]; // Should be 1
    let a3 = net.calc(&[1.0, 1.0])[0]; // Should be 0

    println!("0, 0 -> {} : 0", a0);
    println!("0, 1 -> {} : 1", a1);
    println!("1, 0 -> {} : 1", a2);
    println!("1, 1 -> {} : 0", a3);
}