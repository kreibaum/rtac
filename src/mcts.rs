use crate::game::Game;

/// Implements monte carlo tree search.
pub struct Node<G: Game> {
    pub game: G,
    pub state: G::State,
    pub visit_count: f64,
    pub children: Vec<Edge<G>>,
}

pub struct Edge<G: Game> {
    pub game: G,
    pub action: G::Action,
    pub node: Option<Node<G>>,
    pub visit_count: f64,
    pub total_value: f64,
    pub expected_reward: f64, // Caches visit_count / total_value
    pub prior_probability: f64,
}

impl<G: Game> core::fmt::Debug for Node<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.state)?;
        writeln!(
            f,
            "This node has {} children with a total of {} visits",
            self.children.len(),
            self.visit_count,
        )?;

        for edge in self.children.iter() {
            writeln!(f, "{:?}", edge)?;
        }

        Ok(())
    }
}

impl<G: Game> core::fmt::Debug for Edge<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.node {
            Some(node) => {
                writeln!(
                    f,
                    "Action {:?}, prior probability {:.4}, visit count {}, total value {}, expected reward {}",
                    self.action, self.prior_probability, self.visit_count, self.total_value, self.expected_reward
                )?;
                format!("{:?}", node)
                    .split('\n')
                    .for_each(|line| writeln!(f, "  {}", line).unwrap());
            }
            None => write!(
                f,
                "Action {:?}, prior probability {:.4}, unexpanded node.",
                self.action, self.prior_probability
            )?,
        }
        Ok(())
    }
}

impl<G: Game> Node<G> {
    pub fn new(game: G, state: G::State) -> Node<G> {
        let actions = game.get_actions(&state);

        let prior_probability = 1.0 / actions.len() as f64;

        let children = actions
            .iter()
            .map(|action| Edge::new(game.clone(), *action, prior_probability))
            .collect();

        Node {
            game,
            state,
            visit_count: 0.0,
            children,
        }
    }

    /// Choose an action that maximizes Q+U.
    fn choose_edge_index(&self) -> usize {
        let mut best_action_index = 0;
        let mut best_action_value = -std::f64::INFINITY;

        for (i, action) in self.children.iter().enumerate() {
            let expected_reward = action.expected_reward;
            // We add + 0.0001 so the policy is already respected in the first
            // step.
            let explore_value = self.game.exploration_factor()
                * action.prior_probability
                * ((self.visit_count).sqrt() / (1.0 + action.visit_count) + 0.0001);
            let action_value = expected_reward + explore_value;
            // Note that if two actions have the same value, we always pick
            // the first one.
            if action_value > best_action_value {
                best_action_index = i;
                best_action_value = action_value;
            }
        }

        best_action_index
    }

    pub fn walk_to_leaf(&mut self) -> f64 {
        if self.children.is_empty() {
            return self
                .score_terminal_victory_state(&self.state, self.game.get_player(&self.state));
        }

        let edge_index = self.choose_edge_index();
        let edge = &mut self.children[edge_index];

        let value = if let Some(ref mut child_node) = edge.node {
            // Here we assume alternating players.
            -child_node.walk_to_leaf()
        } else {
            edge.expand(&self.state);

            // Do a random rollout.
            edge.node.as_ref().unwrap().random_rollout()
        };

        edge.total_value += value;
        edge.visit_count += 1.0;
        edge.expected_reward = edge.total_value / edge.visit_count;

        self.visit_count += 1.0;

        value
    }

    pub fn random_rollout(&self) -> f64 {
        let mut state = self.state.clone();
        // We call "random_rollout" after already executing one action.
        // This means the player already changed. We want to evaluate the
        // action from the view of the player who did it and not the
        // player who now has to react to it.
        let player = !self.game.get_player(&state);

        while !self.game.get_victory_state(&state).is_terminal() {
            let actions = self.game.get_actions(&state);
            let action = actions[rand::random::<usize>() % actions.len()];
            self.game.apply_action(&mut state, action);
        }
        self.score_terminal_victory_state(&state, player)
    }

    /// Returns a score for a terminal state. Panics, if the state is not
    /// terminal.
    fn score_terminal_victory_state(
        &self,
        state: &<G as Game>::State,
        player: crate::game::Player,
    ) -> f64 {
        match self.game.get_victory_state(&state) {
            crate::game::VictoryState::InProgress => panic!("Game should be over"),
            crate::game::VictoryState::Draw => 0.0,
            crate::game::VictoryState::Won(winner) => {
                if winner == player {
                    1.0
                } else {
                    -1.0
                }
            }
        }
    }
}

impl<G: Game> Edge<G> {
    fn new(game: G, action: G::Action, prior_probability: f64) -> Edge<G> {
        Edge {
            game,
            action,
            node: None,
            visit_count: 0.0,
            total_value: 0.0,
            expected_reward: 0.0,
            prior_probability,
        }
    }

    fn expand(&mut self, parent_state: &G::State) {
        let mut state = parent_state.clone();
        self.game.apply_action(&mut state, self.action);
        self.node = Some(Node::new(self.game.clone(), state));
    }
}
