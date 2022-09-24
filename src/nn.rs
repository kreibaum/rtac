use dfdx::{
    prelude::{Linear, Module, ReLU, ResetParams, Softmax, SplitInto, Tanh},
    tensor::{HasArrayData, Tensor1D, Tensor3D, TensorCreator},
};

use crate::{
    game::{Game, Player},
    mcts::{Edge, MctsConfigTrait, Node},
    tictactoe::TicTacToe,
};

pub fn tensorize(state: &TicTacToe) -> Tensor1D<9> {
    let mut data = [0.0; 9];

    for i in 0..3 {
        for j in 0..3 {
            let value = match state.board[i][j] {
                None => 0.0,
                Some(Player::X) => 1.0,
                Some(Player::O) => -1.0,
            };
            data[i + j * 3] = value;
        }
    }

    Tensor1D::new(data)
}

/// Neural Network based solution for TicTacToe

#[derive(Debug, Clone)]
pub struct NetworkMctsConfig {
    pub mlp: MLP,
    pub temperature: f32,
}

type MLP = (
    Linear<9, 13>,
    ReLU,
    SplitInto<((Linear<13, 9>, Softmax), (Linear<13, 1>, Tanh))>,
);

impl NetworkMctsConfig {
    pub fn new() -> NetworkMctsConfig {
        let mut rng = rand::thread_rng();

        let mut mlp: MLP = Default::default();
        mlp.reset_params(&mut rng);

        NetworkMctsConfig {
            mlp,
            temperature: 1.0,
        }
    }
}

impl MctsConfigTrait<TicTacToe> for NetworkMctsConfig {
    fn node_for_new_state(&self, state: TicTacToe) -> (crate::mcts::Node<TicTacToe>, f64) {
        let input = tensorize(&state);

        let (policy, value_output) = self.mlp.forward(input);

        let actions = state.get_actions();

        let children = actions
            .iter()
            .map(|action| {
                Edge::new(
                    state.clone(),
                    *action,
                    policy.data()[action.0 + 3 * action.1] as f64,
                )
            })
            .collect();

        let node = Node {
            state,
            visit_count: 0.0,
            children,
        };

        (node, value_output.data()[0] as f64)
    }
}
