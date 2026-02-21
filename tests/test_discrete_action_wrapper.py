import unittest

import sternhalma_v0


class DiscreteActionWrapperTests(unittest.TestCase):
    def test_observe_contains_action_mask_and_wrapped_observation(self):
        env = sternhalma_v0.discrete_action_env(
            num_players=2, board_diagonal=5, render_mode=None, max_actions=16
        )
        env.reset()
        agent = env.agent_selection

        obs = env.observe(agent)
        self.assertIn("observations", obs)
        self.assertIn("action_mask", obs)
        self.assertEqual(obs["action_mask"].shape, (16,))
        self.assertTrue(env.observation_space(agent).contains(obs))
        env.close()

    def test_action_index_maps_to_first_valid_move(self):
        wrapped = sternhalma_v0.discrete_action_env(
            num_players=2, board_diagonal=5, render_mode=None, max_actions=32
        )
        plain = sternhalma_v0.env(num_players=2, board_diagonal=5, render_mode=None)
        wrapped.reset()
        plain.reset()

        acting_agent = wrapped.agent_selection
        first_move = plain.infos[plain.agent_selection]["valid_moves"][0]

        wrapped.step(0)
        plain.step(first_move)

        self.assertEqual(wrapped.unwrapped.state().tolist(), plain.unwrapped.state().tolist())
        self.assertNotEqual(wrapped.agent_selection, acting_agent)
        wrapped.close()
        plain.close()

    def test_invalid_action_index_gets_penalty(self):
        env = sternhalma_v0.discrete_action_env(
            num_players=2, board_diagonal=5, render_mode=None, max_actions=8
        )
        env.reset()
        acting_agent = env.agent_selection

        env.step(999)

        self.assertEqual(env.rewards[acting_agent], -1.0)
        self.assertTrue(env.infos[acting_agent].get("invalid_move", False))
        env.close()


if __name__ == "__main__":
    unittest.main()
