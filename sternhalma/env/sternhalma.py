import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from typing import Dict, Any, List, Tuple, Optional
from gymnasium import spaces

from ..utils.board import Board
from ..utils.types import VariableLengthTupleSpace


class SternhalmaEnvironment(AECEnv):
    """
    A PettingZoo environment for the Sternhalma game (also known as Chinese Checkers).

    Attributes:
        metadata (dict): Metadata for the environment, specifying the render modes available.
        num_players (int): The number of players in the game.
        board (Board): The game board.
        agents (list): List of agents participating in the environment.
        observation_space (spaces.Box): The observation space of the environment, representing the game board.
        action_space (VariableLengthTupleSpace): The action space for the environment, allowing for a range of move
            sequences each turn. Each action is a list of 2D position tuples, representing moves on the board. The space
            is defined with an exponential bias towards shorter move sequences to favor more common game situations.
        rewards (dict): A dictionary mapping agents to their current rewards.
        dones (dict): A dictionary mapping agents to boolean values indicating whether they are done.
        infos (dict): A dictionary mapping agents to additional info dictionaries.
        _agent_selector (agent_selector): A PettingZoo utility to manage turn order among agents.
        agent_selection (str): The currently selected agent.
        agent_name_mapping (dict): A mapping from agent names to their indices.
        char_encoding (dict): Encoding of the board characters for observation space.
    """

    metadata = {'render.modes': ['ansi']}

    def __init__(self, num_players: int, board_diagonal: int):
        """
        Initializes the Sternhalma environment.

        Args:
            num_players (int): Number of players in the game.
            board_diagonal (int): The size of the game board, measured diagonally across.

        Raises:
            ValueError: If the board diagonal is not odd or less than 3, or if an invalid number of players is specified.
        """
        super().__init__()

        self.num_players = num_players
        self.board = Board(board_diagonal, num_players)
        h, w = self.board.get_dims()

        self.agents = [f"player_{i}" for i in range(self.num_players)]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_players))))

        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.char_encoding = {' ': -2, 'O': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, '|': -1}

        self.observation_space = spaces.Box(low=-2, high=6, shape=(h - 1, w - 1), dtype=np.int8)

        # Jumping over all the pieces
        max_length = num_players * ((board_diagonal // 2) * (board_diagonal // 2 + 1)) // 2

        self.action_space = VariableLengthTupleSpace(
            max_length=max_length,
            low=0,
            high=max(h - 1, w - 1)
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> None:
        """
        Resets the environment to its initial state.

        Args:
            seed (Optional[int]): The seed for random number generation.
            options (Optional[Dict]): Additional options for resetting the environment.
        """
        self.board.initialize_board()
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def step(self, action: Dict[str, Any]) -> None:
        agent = self.agent_selection
        player_idx = self.agent_name_mapping[agent]

        move = self.convert_action_to_move(action, agent)
        if self.board.is_valid_move(move, player_idx):
            self.board.make_move(player_idx, move)
            reward = self.calculate_reward(player_idx, move)
            info = self.collect_game_info(player_idx)
        else:
            # For invalid moves
            reward = -1.0  # Penalty for invalid move
            info = {'invalid_move': True}

        self.rewards[agent] += reward
        self.dones[agent] = self.board.check_winner(player_idx) or info.get('invalid_move', False)
        self.infos[agent] = info

    def render(self, mode: str = 'ansi') -> None:
        """
        Renders the current state of the environment.

        Args:
            mode (str): The mode to render with. Currently, only 'ansi' is supported.

        Raises:
            NotImplementedError: If an unsupported render mode is specified.
        """
        if mode == 'ansi':
            self.board.print_board()
        else:
            raise NotImplementedError(f"Render mode {mode} not implemented.")

    def observe(self, agent: str) -> np.ndarray:
        """
        Returns the observation for a given agent.

        Args:
            agent (str): The name of the agent to observe the environment for.

        Returns:
            np.ndarray: The observation of the environment for the specified agent.
        """
        board_array = np.array(self.board.get_grid())
        observation = np.vectorize(self.char_encoding.get)(board_array[1:, 1:])
        return observation

    def num_agents(self) -> int:
        """
        Returns the number of agents in the environment.

        Returns:
            int: The number of agents.
        """
        return self.num_players

    def last(self, observe: bool = True) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Returns the observation, reward, done flag, and info for the last agent selected.

        Args:
            observe (bool): Whether to include the observation in the returned tuple.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: A tuple containing the last observation, reward, done flag, and info for the selected agent.
        """
        agent = self.agent_selection
        observation = self.observe(agent) if observe else None
        reward = self.rewards.get(agent, 0)
        done = self.dones.get(agent, False)
        info = self.infos.get(agent, {})
        return observation, reward, done, info

    def convert_action_to_move(self, action: Dict[str, Any], agent: str) -> List[Tuple[int, int]]:
        """
        Converts an action received from an agent to the corresponding move on the board.

        Args:
            action (Dict[str, Any]): The action to convert.
            agent (str): The name of the agent performing the action.

        Returns:
            List[Tuple[int, int]]: The corresponding move on the board.
        """
        return [(cell[0] + 1, cell[1] + 1) for cell in action[agent]]

    def convert_move_to_action(self, move: List[Tuple[int, int]]) -> Any:
        """
        Converts a move on the board to the corresponding action for an agent.

        Args:
            move (List[Tuple[int, int]]): The move to convert.

        Returns:
            Any: The corresponding action for an agent.
        """
        return [(cell[0] - 1, cell[1] - 1) for cell in move]

    def get_available_actions(self, agent: str) -> List[int]:
        """
        Returns a list of available actions for the specified agent.

        Args:
            agent (str): The name of the agent.

        Returns:
            List[int]: A list of available actions for the agent.
        """
        available_actions = []
        player_idx = self.agent_name_mapping[agent]
        for move in self.board.get_available_moves(player_idx):
            available_actions.append(self.convert_move_to_action(move))
        return available_actions

    def calculate_reward(self, player_idx: int, move: List[Tuple[int, int]]) -> float:
        """
        Calculate the reward for the agent after making a move.
        A reward of 1 is given for each piece that reaches the home triangle.

        Args:
            player_idx (int): The index of the player who made the move.
            move (List[Tuple[int, int]]): The move made by the player, which is a list of tuples representing the positions.

        Returns:
            float: The reward resulting from the move.
        """
        # Check the final position in the move to see if it's in the home triangle
        final_position = move[-1]
        if self.board.is_in_home_triangle(final_position, player_idx):
            return 1.0
        else:
            return 0.0

    def collect_game_info(self, player_idx: int) -> Dict[str, Any]:
        """
        Collect additional information about the game state after an action is taken.

        Args:
            player_idx (int): The index of the player who took the action.

        Returns:
            Dict[str, Any]: A dictionary containing additional game state information.
        """
        info = {
            'winner': self.board.check_winner(player_idx)
        }
        return info


