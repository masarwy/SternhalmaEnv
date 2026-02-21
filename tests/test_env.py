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
        state = env.state()
        expected_shape = env.observation_space(env.agent_selection).shape
        self.assertEqual(state.shape, expected_shape)
        env.close()


if __name__ == "__main__":
    unittest.main()
