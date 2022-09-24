mod game;
mod learning;
mod mcts;
mod nn;
mod tictactoe;

use mcts::MctsConfigTrait;
use tictactoe::TicTacToe;

use crate::nn::NetworkMctsConfig;

fn main() {
    let ttt = TicTacToe::new();

    let config: NetworkMctsConfig = nn::NetworkMctsConfig::new()
        .with_power(10000)
        .with_temperature(1.0)
        .with_batch_size(100);

    let mut node = config.node_for_new_state(ttt).0;

    for _ in 0..10_000 {
        node.walk_to_leaf(&config);
    }

    // Output the values for all actions:
    for edge in node.children.iter() {
        println!(
            "Action {:?} has value {:+.4} and was visited {} times. (Prior: {})",
            edge.action, edge.expected_reward, edge.visit_count, edge.prior_probability
        );
    }

    // Get some training data
    let training_data = learning::generate_training_data(config);
    println!("Training data size: {}", training_data.len());
}
