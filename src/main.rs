mod files;
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
use crate::nn::{tensorize, MultiLayerPerceptron, NetworkMctsConfig};

fn main() {
    // Load Thomas model and apply it to an empty board.
    let mlp_thomas = files::load_model("thomas.mp").unwrap();
    println!("{:#?}", mlp_thomas.0);
    let config_thomas = NetworkMctsConfig {
        mlp: mlp_thomas,
        temperature: 1.0,
        power: 100,
        batch_size: 100,
    };

    let state = TicTacToe::new();
    println!("{:?}", tensorize(&state));
    let (node_thomas, value_thomas) = config_thomas.node_for_new_state(state);
    println!("Value: {}", value_thomas);
    println!("Thomas: \n{:?}", node_thomas);

    let ttt = TicTacToe::new();

    let mut config: NetworkMctsConfig = nn::NetworkMctsConfig::new()
        .with_power(100)
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
    for i_training in 0..2 {
        println!("Training loop {}", i_training);
        // Get some training data
        let training_data: TrainingData<10000> = learning::generate_training_data(&config);
        println!("Training data generated");

        let x = training_data.input;
        let y1 = training_data.improved_policy;
        let y2 = training_data.expected_value;

        let mut sgd: Sgd<MultiLayerPerceptron> = Default::default();

        for i_epoch in 0..2 {
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
    let state = TicTacToe::new();

    model_mcts_example(&config, state.clone());
    plain_mcts_example(state);

    files::save_model(config.mlp.clone(), "model.mp").unwrap();
    let mlp2 = files::load_model("model.mp").unwrap();

    // Check that the layers are equal by comparing data().
    assert_eq!(config.mlp.0.weight.data(), mlp2.0.weight.data());
    files::save_model(mlp2, "model2.mp").unwrap();
}

fn model_mcts_example(config: &NetworkMctsConfig, mut state: TicTacToe) {
    // Model intuition after training:
    state.apply_action((0, 0));
    println!();
    println!("{}", state);
    println!("Testing mcts after training");
    let mut node = config.node_for_new_state(state.clone()).0;
    for _ in 0..100 {
        node.walk_to_leaf(config);
    }
    // Output the values for all actions:
    for edge in node.children.iter() {
        println!(
            "Action {:?} has value {:+.4} and was visited {} times. (Prior: {})",
            edge.action, edge.expected_reward, edge.visit_count, edge.prior_probability
        );
    }
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
