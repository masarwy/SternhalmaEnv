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


if __name__ == "__main__":
    unittest.main()
