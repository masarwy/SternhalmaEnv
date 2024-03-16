import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from typing import Dict, Any, List, Tuple, Optional
from gymnasium import spaces

from ..utils.board import Board


class SternhalmaEnvironment(AECEnv):
    metadata = {'render.modes': ['ansi']}

    def __init__(self, num_players: int, board_diagonal: int):
        # Check if the diagonal is odd and >= 3
        if board_diagonal < 3 or board_diagonal % 2 == 0:
            raise ValueError("Board diagonal must be an odd number and greater than or equal to 3.")

        # Check if the number of players is valid
        if num_players not in [2, 3, 4, 6]:
            raise ValueError("Number of players must be 2, 3, 4, or 6.")
        super().__init__()

        self.num_players = num_players

        # Initialize the board
        self.board = Board(board_diagonal, num_players)
        h, w = self.board.get_dims()

        # Setup for PettingZoo
        self.agents = [f"player_{i}" for i in range(self.num_players)]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_players))))

        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Define encoding for the observation space
        self.char_encoding = {' ': -2, 'O': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, '|': -1}

        self.observation_space = spaces.Box(low=-2, high=6, shape=(h - 1, w - 1), dtype=np.int8)

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> None:
        self.board.initialize_board()  # Assuming your Board class has this method
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def step(self, action: Dict[str, Any]) -> None:
        agent = self.agent_selection
        player_idx = self.agent_name_mapping[agent]

        # Convert the action to the form expected by your Board class
        move = self.convert_action_to_move(action, agent)

        if self.board.is_valid_move(move, player_idx):
            self.board.make_move(player_idx, move)

            # Calculate rewards, set done flags, and collect any additional info
            self.rewards[agent] += self.calculate_reward(player_idx)
            self.dones[agent] = self.board.check_winner(player_idx)
            self.infos[agent] = self.collect_game_info(player_idx)

        self.agent_selection = self._agent_selector.next()  # Get the next agent

    def render(self, mode: str = 'ansi') -> None:
        if mode == 'ansi':
            self.board.print_board()
        else:
            # Raise an error for modes that are not implemented
            raise NotImplementedError(f"Render mode {mode} not implemented.")

    def observe(self, agent: str) -> Any:
        board_array = np.array(self.board.get_grid())
        observation = np.vectorize(self.char_encoding.get)(board_array[1:, 1:])
        return observation

    def num_agents(self) -> int:
        # Return the number of agents
        return self.num_players

    def last(self, observe: bool = True) -> Tuple[Any, float, bool, Dict[str, Any]]:
        agent = self.agent_selection
        observation = self.observe(agent)  # Get the last observation
        reward = self.rewards.get(agent, 0)
        done = self.dones.get(agent, False)
        info = self.infos.get(agent, {})
        return observation, reward, done, info

    def convert_action_to_move(self, action: Dict[str, Any], agent: str) -> List[tuple[int, int]]:
        return [(cell[0] + 1, cell[1] + 1) for cell in action[agent]]
