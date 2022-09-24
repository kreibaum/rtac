mod game;
mod mcts;
mod tictactoe;

use tictactoe::TicTacToe;

use crate::mcts::Node;

fn main() {
    let ttt = TicTacToe::new();

    let mut node = Node::new(ttt);

    for _ in 0..10_000 {
        node.walk_to_leaf();
    }

    // Output the values for all actions:
    for edge in node.children.iter() {
        println!(
            "Action {:?} has value {:+.4} and was visited {} times",
            edge.action, edge.expected_reward, edge.visit_count
        );
    }
}
