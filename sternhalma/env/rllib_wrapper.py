from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper


class DiscreteActionMaskWrapper(BaseWrapper):
    """
    Wraps the Sternhalma AEC env with a fixed-size Discrete action space and action masks.

    Observation format:
    {
        "observations": {"board": ..., "current_player": ...},
        "action_mask": np.ndarray shape=(max_actions,), dtype=np.int8
    }
    """

    def __init__(self, env, max_actions: int = 256):
        super().__init__(env)
        if max_actions <= 0:
            raise ValueError("max_actions must be > 0")
        self.max_actions = max_actions

        self._action_spaces = {
            agent: spaces.Discrete(self.max_actions) for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    "observations": env.observation_space(agent),
                    "action_mask": spaces.MultiBinary(self.max_actions),
                }
            )
            for agent in self.possible_agents
        }

    def action_space(self, agent):
        return self._action_spaces[agent]

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @staticmethod
    def _normalize_move(move: Any) -> Optional[Tuple[Tuple[int, int], ...]]:
        if not isinstance(move, list):
            return None
        normalized: List[Tuple[int, int]] = []
        for cell in move:
            if not isinstance(cell, (tuple, list)) or len(cell) != 2:
                return None
            if not isinstance(cell[0], (int, np.integer)) or not isinstance(cell[1], (int, np.integer)):
                return None
            normalized.append((int(cell[0]), int(cell[1])))
        return tuple(normalized)

    def _indexed_valid_moves(self, agent: str) -> List[List[Tuple[int, int]]]:
        valid_moves = self.infos.get(agent, {}).get("valid_moves", [])
        unique_moves: List[List[Tuple[int, int]]] = []
        seen = set()
        for move in valid_moves:
            key = self._normalize_move(move)
            if key is None or key in seen:
                continue
            seen.add(key)
            unique_moves.append(move)
            if len(unique_moves) >= self.max_actions:
                break
        return unique_moves

    def observe(self, agent):
        base_obs = self.env.observe(agent)
        mask = np.zeros(self.max_actions, dtype=np.int8)

        if agent == self.agent_selection and not self.terminations[agent] and not self.truncations[agent]:
            indexed_moves = self._indexed_valid_moves(agent)
            mask[: len(indexed_moves)] = 1

        return {"observations": base_obs, "action_mask": mask}

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return super().step(action)

        agent = self.agent_selection
        indexed_moves = self._indexed_valid_moves(agent)

        if not indexed_moves:
            return super().step([])

        if isinstance(action, (int, np.integer)) and 0 <= int(action) < len(indexed_moves):
            return super().step(indexed_moves[int(action)])

        # Send a shape-valid but illegal action so underlying env applies invalid-action penalty.
        return super().step([(0, 0)])
