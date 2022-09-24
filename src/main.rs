mod game;
mod mcts;
mod tictactoe;

use game::Game;
use tictactoe::TicTacToeGame;

use crate::mcts::Node;

fn main() {
    println!("Hello, world!");

    let ttt = TicTacToeGame.initial_state();

    let mut node = Node::new(TicTacToeGame, ttt);

    for i in 0..10000 {
        node.walk_to_leaf();
    }

    // Output the values for all actions:
    for edge in node.children.iter() {
        println!(
            "Action {:?} has value {}",
            edge.action, edge.expected_reward
        );
    }

    //println!("Node: {:#?}, value: {}", node, value);
}
