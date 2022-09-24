mod game;
mod mcts;
mod tictactoe;

use mcts::{MctsConfigTrait, RolloutMctsConfig};
use tictactoe::TicTacToe;

fn main() {
    let ttt = TicTacToe::new();

    let config: RolloutMctsConfig<TicTacToe> = RolloutMctsConfig::default();

    let mut node = config.node_for_new_state(ttt).0;

    for _ in 0..10_000 {
        node.walk_to_leaf(config);
    }

    // Output the values for all actions:
    for edge in node.children.iter() {
        println!(
            "Action {:?} has value {:+.4} and was visited {} times",
            edge.action, edge.expected_reward, edge.visit_count
        );
    }
}
