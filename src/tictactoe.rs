use std::fmt::Display;

use crate::game::{Game, Player};

/// Implements a simple TicTacToe game.

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TicTacToe {
    board: [[Option<Player>; 3]; 3],
    current_player: Player,
}

type Action = (usize, usize);

impl TicTacToe {
    fn new() -> TicTacToe {
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

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TicTacToeGame;

impl Game for TicTacToeGame {
    type Action = Action;
    type State = TicTacToe;

    fn initial_state(&self) -> Self::State {
        TicTacToe::new()
    }

    fn get_actions(&self, state: &Self::State) -> Vec<Self::Action> {
        let mut result = Vec::with_capacity(9);
        for x in 0..3 {
            for y in 0..3 {
                if state.board[x][y].is_none() {
                    result.push((x, y));
                }
            }
        }
        result
    }

    fn apply_action(&self, state: &mut Self::State, action: Self::Action) {
        state.play(self.get_player(state), action);
        state.current_player = !state.current_player;
    }

    fn get_victory_state(&self, state: &Self::State) -> crate::game::VictoryState {
        if let Some(winner) = state.winner() {
            crate::game::VictoryState::Won(winner)
        } else if self.get_actions(state).is_empty() {
            crate::game::VictoryState::Draw
        } else {
            crate::game::VictoryState::InProgress
        }
    }

    fn get_player(&self, state: &Self::State) -> Player {
        state.current_player
    }
}
