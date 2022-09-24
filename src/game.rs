/// Defines a trait for generic games on which we can run MCTS.
use std::fmt::Debug;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Player {
    X,
    O,
}

pub enum VictoryState {
    InProgress,
    Draw,
    Won(Player),
}

impl VictoryState {
    pub fn is_terminal(&self) -> bool {
        match self {
            VictoryState::InProgress => false,
            _ => true,
        }
    }
}

pub trait Game: Clone + Debug {
    type Action: Copy + Debug;
    type State: Clone + Debug;

    fn initial_state(&self) -> Self::State;
    fn get_actions(&self, state: &Self::State) -> Vec<Self::Action>;

    /// Applies an action to a state and returns the new state.
    fn apply_action(&self, state: &mut Self::State, action: Self::Action);
    fn get_victory_state(&self, state: &Self::State) -> VictoryState;
    fn get_player(&self, state: &Self::State) -> Player;

    fn exploration_factor(&self) -> f64 {
        1.0
    }
}
