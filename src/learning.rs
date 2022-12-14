use dfdx::tensor::{HasArrayData, Tensor1D, Tensor2D, TensorCreator};

use crate::{
    game::Game,
    mcts::MctsConfigTrait,
    nn::{tensorize, NetworkMctsConfig},
    tictactoe::{action_to_index, index_to_action, TicTacToe},
};

#[derive(Debug)]
pub struct TrainingDatum {
    pub input: Tensor1D<9>,
    pub improved_policy: Tensor1D<9>,
    pub expected_value: f32,
}

#[derive(Debug)]
pub struct TrainingData<const N: usize> {
    pub input: Tensor2D<N, 9>,
    pub improved_policy: Tensor2D<N, 9>,
    pub expected_value: Tensor2D<N, 1>,
}

impl<const N: usize> TrainingData<N> {
    pub fn new(data: &[TrainingDatum]) -> Self {
        let mut input = Tensor2D::zeros();
        let mut improved_policy = Tensor2D::zeros();
        let mut expected_value = Tensor2D::zeros();

        for (i, datum) in data.iter().enumerate() {
            input.mut_data()[i].copy_from_slice(datum.input.data());
            improved_policy.mut_data()[i].copy_from_slice(datum.improved_policy.data());
            expected_value.mut_data()[i][0] = datum.expected_value;
        }

        TrainingData {
            input,
            improved_policy,
            expected_value,
        }
    }
}

fn one_training_step(state: &TicTacToe, config: NetworkMctsConfig) -> Option<TrainingDatum> {
    // If the state is already terminal, there is no point in training on it.
    if state.get_victory_state().is_terminal() {
        return None;
    }

    let mut node = config.node_for_new_state(state.clone()).0;

    for _ in 0..config.power {
        node.walk_to_leaf(&config);
    }

    // We want to apply the softmax to the expected reward to determine the
    // improved policy.

    let mut output: [f32; 9] = [0.0; 9];
    let mut sum: f32 = 0.0;
    let mut best_value = f32::NEG_INFINITY;

    // Get maximum visit count to prevent numeric problems with the softmax.
    let max_visit_count = node
        .children
        .iter()
        .map(|c| c.visit_count)
        // .max() does not work, because f32 isn't Eq.
        .max_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap())
        .unwrap();

    for edge in node.children.iter() {
        let value = ((edge.visit_count - max_visit_count) / config.temperature).exp();
        output[action_to_index(edge.action)] = value;
        sum += value;
        if value > best_value {
            best_value = edge.expected_reward;
        }
    }
    output.iter_mut().for_each(|x| *x /= sum);

    let improved_policy = Tensor1D::new(output);

    Some(TrainingDatum {
        input: tensorize(state),
        improved_policy,
        expected_value: best_value,
    })
}

pub fn generate_training_data<const N: usize>(config: &NetworkMctsConfig) -> TrainingData<N> {
    let mut data = Vec::with_capacity(N);

    // Collect training data.
    while data.len() < config.batch_size {
        // Play out one game driven by the training:
        let mut state = TicTacToe::new();
        while !state.get_victory_state().is_terminal() && data.len() < config.batch_size {
            let datum = one_training_step(&state, config.clone());
            if let Some(datum) = datum {
                let random_distribution = datum.improved_policy.data();
                let action_index = sample_index_from_distribution(random_distribution);
                let action = index_to_action(action_index);
                state.apply_action(action);

                data.push(datum);
            } else {
                panic!("Didn't get any training data from a non-terminal state.");
            }
        }
    }

    TrainingData::new(&data)
}

fn sample_index_from_distribution(random_distribution: &[f32]) -> usize {
    // Sample an action from the posterior policy:
    // This means we pull a random index weighted by datum.improved_policy.
    let random_float = rand::random::<f32>();
    let mut cumulative_probability = 0.0;
    let mut action_index = 0;
    for (i, probability) in random_distribution.iter().enumerate() {
        cumulative_probability += probability;
        if cumulative_probability > random_float {
            action_index = i;
            break;
        }
    }
    action_index
}
