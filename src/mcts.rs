use crate::game::Game;

// Name sucks, needs to be changed
pub trait MctsConfigTrait<G: Game>: Copy {
    // Returns a new node as well as the estimated value of the node.
    // This new node than already contains all the children with their
    // prior values.
    fn node_for_new_state(&self, state: G) -> (Node<G>, f64);
}

#[derive(Debug, Clone)]
pub struct RolloutMctsConfig<G: Game> {
    phantom_data: std::marker::PhantomData<G>,
}

// I can't derive this due to the phantom data.
impl<G: Game> Default for RolloutMctsConfig<G> {
    fn default() -> Self {
        Self {
            phantom_data: std::marker::PhantomData,
        }
    }
}

impl<G: Game> Copy for RolloutMctsConfig<G> {}

impl<G: Game> MctsConfigTrait<G> for RolloutMctsConfig<G> {
    fn node_for_new_state(&self, state: G) -> (Node<G>, f64) {
        let actions = state.get_actions();

        let prior_probability = 1.0 / actions.len() as f64;

        let children = actions
            .iter()
            .map(|action| {
                Edge::new(
                    state.clone(),
                    *action,
                    prior_probability + 0.001 * rand::random::<f64>(),
                )
            })
            .collect();

        let node = Node {
            state,
            visit_count: 0.0,
            children,
        };

        let mut state_clone = node.state.clone();
        random_rollout(&mut state_clone);
        let value = score_terminal_victory_state(&state_clone, node.state.get_player());

        (node, value)
    }
}

/// Implements monte carlo tree search.
pub struct Node<G: Game> {
    pub state: G,
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
    /// Choose an action that maximizes Q+U.
    fn choose_edge_index(&self) -> usize {
        let mut best_action_index = 0;
        let mut best_action_value = -std::f64::INFINITY;

        for (i, action) in self.children.iter().enumerate() {
            let expected_reward = action.expected_reward;
            // We add + 0.0001 so the policy is already respected in the first
            // step.
            let explore_value = self.state.exploration_factor()
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

    pub fn walk_to_leaf(&mut self, config: impl MctsConfigTrait<G>) -> f64 {
        if self.children.is_empty() {
            return score_terminal_victory_state(&self.state, self.state.get_player());
        }

        let edge_index = self.choose_edge_index();
        let edge = &mut self.children[edge_index];

        let value = if let Some(ref mut child_node) = edge.node {
            // Here we assume alternating players.
            -child_node.walk_to_leaf(config)
        } else {
            let mut new_state = self.state.clone();
            new_state.apply_action(edge.action);
            let (new_node, value) = config.node_for_new_state(new_state);

            edge.node = Some(new_node);
            -value
        };

        edge.total_value += value;
        edge.visit_count += 1.0;
        edge.expected_reward = edge.total_value / edge.visit_count;

        self.visit_count += 1.0;

        value
    }

    pub fn random_rollout(&self) -> f64 {
        // We call "random_rollout" after already executing one action.
        // This means the player already changed. We want to evaluate the
        // action from the view of the player who did it and not the
        // player who now has to react to it.
        let player = !self.state.get_player();

        let mut state = self.state.clone();
        random_rollout(&mut state);
        score_terminal_victory_state(&state, player)
    }
}

fn random_rollout<G: Game>(state: &mut G) {
    while !state.get_victory_state().is_terminal() {
        let actions = state.get_actions();
        let action = actions[rand::random::<usize>() % actions.len()];
        state.apply_action(action);
    }
}

/// Returns a score for a terminal state. Panics, if the state is not
/// terminal.
fn score_terminal_victory_state(state: &impl Game, player: crate::game::Player) -> f64 {
    match state.get_victory_state() {
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
}
