/// Defines a trait for generic games on which we can run MCTS.
use std::{
    fmt::{Debug, Display},
    ops::Not,
};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Player {
    X,
    O,
}

impl Not for Player {
    type Output = Player;

    fn not(self) -> Self::Output {
        match self {
            Player::X => Player::O,
            Player::O => Player::X,
        }
    }
}

#[derive(Debug)]
pub enum VictoryState {
    InProgress,
    Draw,
    Won(Player),
}

impl VictoryState {
    pub fn is_terminal(&self) -> bool {
        !matches!(self, VictoryState::InProgress)
    }
}

pub trait Game: Clone + Debug + Display {
    type Action: Copy + Debug;

    fn get_actions(&self) -> Vec<Self::Action>;

    /// Applies an action to a state, mutating it.
    fn apply_action(&mut self, action: Self::Action);
    fn get_victory_state(&self) -> VictoryState;
    fn get_player(&self) -> Player;

    fn exploration_factor(&self) -> f32 {
        1.4
    }
}

// pub trait GameMetadata {
//     type Game: Game;
//     fn initial_state() -> Self::Game;
// }
