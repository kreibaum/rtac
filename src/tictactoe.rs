use std::fmt::Display;

use crate::game::{Game, Player};

/// Implements a simple TicTacToe game.

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TicTacToe {
    pub board: [[Option<Player>; 3]; 3],
    pub current_player: Player,
}

type Action = (usize, usize);

pub fn action_to_index(action: Action) -> usize {
    action.0 + action.1 * 3
}
pub fn index_to_action(index: usize) -> Action {
    (index % 3, index / 3)
}

impl TicTacToe {
    pub fn new() -> TicTacToe {
        TicTacToe {
            board: [[None, None, None], [None, None, None], [None, None, None]],
            current_player: Player::X,
        }
    }

    fn play(&mut self, player: Player, (x, y): Action) -> bool {
        if self.board[x][y].is_some() {
            return false;
        }

        self.board[x][y] = Some(player);
        true
    }

    fn winner(&self) -> Option<Player> {
        let mut winner = None;

        for i in 0..3 {
            if self.board[i][0] == self.board[i][1] && self.board[i][1] == self.board[i][2] {
                winner = self.board[i][0];
            }

            if self.board[0][i] == self.board[1][i] && self.board[1][i] == self.board[2][i] {
                winner = self.board[0][i];
            }
        }

        if self.board[0][0] == self.board[1][1] && self.board[1][1] == self.board[2][2] {
            winner = self.board[0][0];
        }

        if self.board[0][2] == self.board[1][1] && self.board[1][1] == self.board[2][0] {
            winner = self.board[0][2];
        }

        winner
    }
}

impl Display for TicTacToe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..3 {
            for j in 0..3 {
                match self.board[i][j] {
                    Some(Player::X) => write!(f, "X")?,
                    Some(Player::O) => write!(f, "O")?,
                    None => write!(f, "•")?,
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl Game for TicTacToe {
    type Action = Action;

    fn get_actions(&self) -> Vec<Self::Action> {
        let mut result = Vec::with_capacity(9);
        for x in 0..3 {
            for y in 0..3 {
                if self.board[x][y].is_none() {
                    result.push((x, y));
                }
            }
        }
        result
    }

    fn apply_action(&mut self, action: Self::Action) {
        self.play(self.get_player(), action);
        self.current_player = !self.current_player;
    }

    fn get_victory_state(&self) -> crate::game::VictoryState {
        if let Some(winner) = self.winner() {
            crate::game::VictoryState::Won(winner)
        } else if self.get_actions().is_empty() {
            crate::game::VictoryState::Draw
        } else {
            crate::game::VictoryState::InProgress
        }
    }

    fn get_player(&self) -> Player {
        self.current_player
    }
}
