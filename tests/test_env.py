import unittest

try:
    import sternhalma_v0
except ModuleNotFoundError as exc:
    sternhalma_v0 = None
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = ""


@unittest.skipIf(sternhalma_v0 is None, f"env dependencies are not installed: {IMPORT_ERROR}")
class EnvTests(unittest.TestCase):
    def test_default_reward_mode_is_sparse(self):
        env = sternhalma_v0.env(num_players=2, board_diagonal=5, render_mode=None)
        self.assertEqual(env.unwrapped.reward_mode, "sparse")
        env.close()

    def test_invalid_reward_mode_raises(self):
        with self.assertRaises(ValueError):
            sternhalma_v0.env(
                num_players=2,
                board_diagonal=5,
                render_mode=None,
                reward_mode="invalid_mode",
            )

    def test_invalid_gamma_raises(self):
        with self.assertRaises(ValueError):
            sternhalma_v0.env(
                num_players=2,
                board_diagonal=5,
                render_mode=None,
                reward_mode="potential_shaped",
                gamma=1.5,
            )

    def test_noop_action_advances_turn(self):
        env = sternhalma_v0.env(num_players=2, board_diagonal=5, render_mode=None)
        env.reset()
        first_agent = env.agent_selection

        env.step(None)
        self.assertNotEqual(env.agent_selection, first_agent)
        env.close()

    def test_invalid_action_gets_penalty_and_sets_invalid_move_info(self):
        env = sternhalma_v0.env(num_players=2, board_diagonal=5, render_mode=None)
        env.reset()
        acting_agent = env.agent_selection

        env.step([(0, 0)])  # invalid: move path length < 2 after conversion

        self.assertEqual(env.rewards[acting_agent], -1.0)
        self.assertTrue(env.infos[acting_agent].get("invalid_move", False))
        env.close()

    def test_state_matches_observation_space_shape(self):
        env = sternhalma_v0.env(num_players=2, board_diagonal=5, render_mode=None)
        env.reset()
        observation = env.observe(env.agent_selection)
        state = env.state()

        self.assertIn("board", observation)
        self.assertIn("current_player", observation)
        self.assertEqual(observation["board"].shape, state.shape)
        self.assertEqual(int(observation["current_player"]), env.agents.index(env.agent_selection))
        self.assertTrue(env.observation_space(env.agent_selection).contains(observation))
        env.close()

    def test_step_rejects_action_not_in_valid_moves(self):
        env = sternhalma_v0.env(num_players=2, board_diagonal=5, render_mode=None)
        env.reset()
        acting_agent = env.agent_selection
        invalid_action = [(0, 0), (0, 1)]

        self.assertNotIn(invalid_action, env.infos[acting_agent]["valid_moves"])

        original_is_valid_move = env.unwrapped.board.is_valid_move
        env.unwrapped.board.is_valid_move = lambda _move, _player_idx: True
        try:
            env.step(invalid_action)
        finally:
            env.unwrapped.board.is_valid_move = original_is_valid_move

        self.assertEqual(env.rewards[acting_agent], -1.0)
        self.assertTrue(env.infos[acting_agent].get("invalid_move", False))
        env.close()

    def test_dead_agent_requires_none_passthrough(self):
        env = sternhalma_v0.env(num_players=2, board_diagonal=5, render_mode=None)
        env.reset()
        dead_agent = env.agent_selection

        env.unwrapped.terminations[dead_agent] = True
        env.step(None)

        self.assertNotIn(dead_agent, env.agents)
        env.close()

    def test_reset_rebuilds_rewards_after_dead_agent_removal(self):
        env = sternhalma_v0.env(num_players=2, board_diagonal=5, render_mode=None)
        env.reset()
        dead_agent = env.agent_selection

        env.unwrapped.terminations[dead_agent] = True
        env.step(None)
        self.assertNotIn(dead_agent, env.rewards)

        env.reset()

        self.assertEqual(set(env.rewards.keys()), set(env.possible_agents))
        self.assertTrue(all(value == 0.0 for value in env.rewards.values()))
        env.close()

    def test_dense_reward_uses_distance_progress(self):
        env = sternhalma_v0.env(num_players=2, board_diagonal=5, render_mode=None, reward_mode="dense")
        env.reset()
        raw = env.unwrapped
        move = [(1, 1), (2, 2)]
        distances = {(1, 1): 5, (2, 2): 3}
        original_distance = raw._distance_to_home
        raw._distance_to_home = lambda pos, _player_idx: distances[pos]
        try:
            self.assertEqual(raw.calculate_reward(0, move), 2.0)
        finally:
            raw._distance_to_home = original_distance
            env.close()

    def test_potential_shaped_reward_adds_sparse_and_distance_progress(self):
        env = sternhalma_v0.env(
            num_players=2,
            board_diagonal=5,
            render_mode=None,
            reward_mode="potential_shaped",
        )
        env.reset()
        raw = env.unwrapped
        start_position = (1, 1)
        final_position = (2, 2)
        move = [start_position, final_position]
        distances = {start_position: 4, final_position: 1}
        original_distance = raw._distance_to_home
        original_in_home = raw.board.is_in_home_triangle
        raw._distance_to_home = lambda pos, _player_idx: distances[pos]
        raw.board.is_in_home_triangle = lambda pos, _player_idx: pos == final_position
        try:
            self.assertEqual(raw.calculate_reward(0, move), 4.0)
        finally:
            raw._distance_to_home = original_distance
            raw.board.is_in_home_triangle = original_in_home
            env.close()

    def test_potential_shaped_reward_uses_custom_gamma(self):
        env = sternhalma_v0.env(
            num_players=2,
            board_diagonal=5,
            render_mode=None,
            reward_mode="potential_shaped",
            gamma=0.5,
        )
        env.reset()
        raw = env.unwrapped
        move = [(1, 1), (2, 2)]
        distances = {(1, 1): 4, (2, 2): 1}
        original_distance = raw._distance_to_home
        original_in_home = raw.board.is_in_home_triangle
        raw._distance_to_home = lambda pos, _player_idx: distances[pos]
        raw.board.is_in_home_triangle = lambda _pos, _player_idx: False
        try:
            self.assertEqual(raw.calculate_reward(0, move, raw.gamma), 3.5)
        finally:
            raw._distance_to_home = original_distance
            raw.board.is_in_home_triangle = original_in_home
            env.close()


if __name__ == "__main__":
    unittest.main()
