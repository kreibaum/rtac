mod game;
mod mcts;
mod nn;
mod tictactoe;

use mcts::MctsConfigTrait;
use tictactoe::TicTacToe;

use crate::nn::NetworkMctsConfig;

fn main() {
    let ttt = TicTacToe::new();

    let config: NetworkMctsConfig = nn::NetworkMctsConfig::new();

    let mut node = config.node_for_new_state(ttt).0;

    for _ in 0..100_000 {
        node.walk_to_leaf(&config);
    }

    // Output the values for all actions:
    for edge in node.children.iter() {
        println!(
            "Action {:?} has value {:+.4} and was visited {} times. (Prior: {})",
            edge.action, edge.expected_reward, edge.visit_count, edge.prior_probability
        );
    }
}
