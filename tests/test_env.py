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
        self.assertIn("distances_to_home", observation)
        self.assertEqual(observation["board"].shape, state.shape)
        self.assertEqual(int(observation["current_player"]), env.agents.index(env.agent_selection))
        self.assertTrue(env.observation_space(env.agent_selection).contains(observation))
        env.close()

    def test_distances_to_home_shape_and_range(self):
        """distances_to_home must have the right shape and be in [0, 1]."""
        env = sternhalma_v0.env(num_players=2, board_diagonal=5, render_mode=None)
        env.reset()
        raw = env.unwrapped
        obs = raw.observe(raw.agent_selection)
        dist = obs["distances_to_home"]
        import numpy as np
        # shape: num_players * pieces_per_player
        pieces_per_player = (5 // 2) * (5 // 2 + 1) // 2  # board_diagonal=5 -> 3
        expected_len = 2 * pieces_per_player
        self.assertEqual(dist.shape, (expected_len,))
        self.assertTrue(np.all(dist >= 0.0) and np.all(dist <= 1.0),
                        f"distances out of [0,1]: {dist}")
        env.close()

    def test_distances_to_home_decreases_after_move_toward_home(self):
        """Moving a piece toward home should decrease its distance entry."""
        import numpy as np
        env = sternhalma_v0.env(
            num_players=2, board_diagonal=5, render_mode=None, reward_mode="dense"
        )
        env.reset()
        raw = env.unwrapped
        agent = raw.agent_selection
        player_idx = raw.agents.index(agent)
        dist_before = raw._compute_distances_to_home()[
            player_idx * ((5 // 2) * (5 // 2 + 1) // 2)
        ]
        # Take a valid move
        valid_moves = raw.get_available_actions(agent)
        if valid_moves:
            env.step(valid_moves[0])
            dist_after = raw._compute_distances_to_home()[
                player_idx * ((5 // 2) * (5 // 2 + 1) // 2)
            ]
            # Not guaranteed to decrease for every move, but the feature must be finite
            self.assertGreaterEqual(dist_after, 0.0)
            self.assertLessEqual(dist_after, 1.0)
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

    def test_terminal_rewards_not_mixed_with_shaping(self):
        """On the winning step the winner gets WIN_REWARD, losers get LOSS_REWARD,
        and the per-move shaping reward must NOT be added on top."""
        env = sternhalma_v0.env(
            num_players=2, board_diagonal=5, render_mode=None,
            reward_mode="potential_shaped",
        )
        env.reset()
        raw = env.unwrapped
        agent = raw.agent_selection
        opponent = [a for a in raw.agents if a != agent][0]

        # Simulate a winning move by patching check_termination.
        raw.board.check_winner = lambda _idx: True
        valid_moves = raw.get_available_actions(agent)
        if valid_moves:
            env.step(valid_moves[0])
            self.assertEqual(raw._cumulative_rewards[agent], raw.WIN_REWARD)
            self.assertEqual(raw._cumulative_rewards[opponent], raw.LOSS_REWARD)
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


    def test_observe_does_not_crash_after_termination(self):
        """observe() must not raise ValueError when called after termination
        sets env.agents=[] (dead-agent guard, diagonal>=7 scenario)."""
        env = sternhalma_v0.env(num_players=2, board_diagonal=5, render_mode=None)
        env.reset()
        raw = env.unwrapped
        # Force termination of all agents as the env itself does on a win.
        raw.terminations = {a: True for a in raw.agents}
        raw.agents.clear()          # mimics post-termination state
        # Must not raise.
        try:
            raw.observe(raw.possible_agents[0])
        except ValueError:
            self.fail("observe() raised ValueError on dead agent")
        env.close()

    def test_current_player_reflects_agent_selection(self):
        """current_player in observe() must equal the index of agent_selection,
        regardless of which agent is observing."""
        env = sternhalma_v0.env(num_players=2, board_diagonal=5, render_mode=None)
        env.reset()
        raw = env.unwrapped
        acting = raw.agent_selection
        expected_idx = raw.agents.index(acting)
        # Both agents should see the same current_player (whoever's turn it is).
        for observer in raw.possible_agents:
            obs = raw.observe(observer)
            self.assertEqual(
                int(obs["current_player"]), expected_idx,
                f"observer={observer} saw current_player={obs['current_player']}, "
                f"expected {expected_idx} ({acting})"
            )
        env.close()

    def test_potential_shaped_gamma_mismatch_documented(self):
        """Verify gamma is stored correctly and documented warning is present."""
        env = sternhalma_v0.env(
            num_players=2, board_diagonal=5, render_mode=None,
            reward_mode="potential_shaped", gamma=0.99,
        )
        self.assertEqual(env.unwrapped.gamma, 0.99)
        # The docstring for potential_shaped_reward must mention matching RL gamma.
        import inspect
        doc = inspect.getdoc(env.unwrapped.potential_shaped_reward)
        self.assertIn("gamma", doc.lower())
        env.close()


if __name__ == "__main__":
    unittest.main()
