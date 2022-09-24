mod game;
mod learning;
mod mcts;
mod nn;
mod tictactoe;

use std::time::Instant;

use dfdx::prelude::{Optimizer, Sgd};
use dfdx::tensor::{HasArrayData, PutTape};
use dfdx::{
    prelude::{mse_loss, Module},
    tensor::Tensor,
};
use mcts::MctsConfigTrait;
use tictactoe::TicTacToe;

use crate::game::Game;
use crate::learning::TrainingData;
use crate::mcts::RolloutMctsConfig;
use crate::nn::{MultiLayerPerceptron, NetworkMctsConfig};

fn main() {
    let ttt = TicTacToe::new();

    let mut config: NetworkMctsConfig = nn::NetworkMctsConfig::new()
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

    // Do a few training loops
    for i_training in 0..10 {
        println!("Training loop {}", i_training);
        // Get some training data
        let training_data: TrainingData<200> = learning::generate_training_data(&config);
        println!("Training data generated");

        let x = training_data.input;
        let y1 = training_data.improved_policy;
        let y2 = training_data.expected_value;

        let mut sgd: Sgd<MultiLayerPerceptron> = Default::default();

        for i_epoch in 0..20 {
            // Train one epoch
            let start = Instant::now();
            let x = x.trace();
            let (pred1, pred2) = config.mlp.forward(x);

            // NOTE: we also have to move the tape around when computing losses
            let (loss2, tape) = mse_loss(pred2, &y2).split_tape();
            // TODO: This should use some cross entropy loss function
            let loss1 = mse_loss(pred1.put_tape(tape), &y1);

            let losses = [*loss1.data(), *loss2.data()];
            let loss = loss1 + &loss2;
            let gradients = loss.backward();
            sgd.update(&mut config.mlp, gradients)
                .expect("Unused params");

            println!(
                "losses={:.3?} in {:?} -- epoch {}",
                losses,
                start.elapsed(),
                i_epoch
            );
        }
    }

    // Model intuition after training:
    let mut state = TicTacToe::new();
    state.apply_action((0, 0));
    println!();
    println!("{}", state);
    println!("Testing mcts after training");
    let mut node = config.node_for_new_state(state.clone()).0;
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

    plain_mcts_example(state);
}

fn plain_mcts_example(state: TicTacToe) {
    // Compare to MCTS
    println!();
    println!("Comparing with plain MCTS");
    let config2: RolloutMctsConfig<TicTacToe> = Default::default();
    let mut node = config2.node_for_new_state(state).0;
    for _ in 0..10_000 {
        node.walk_to_leaf(&config2);
    }
    // Output the values for all actions:
    for edge in node.children.iter() {
        println!(
            "Action {:?} has value {:+.4} and was visited {} times. (Prior: {})",
            edge.action, edge.expected_reward, edge.visit_count, edge.prior_probability
        );
    }
}
