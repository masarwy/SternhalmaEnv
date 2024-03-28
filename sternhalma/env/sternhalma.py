import math
import numpy as np
import gymnasium
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from typing import Dict, Any, List, Tuple, Optional
from gymnasium import spaces
import pygame

from ..utils.board import Board
from ..utils.types import VariableLengthTupleSpace


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
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
        infos (dict): A dictionary mapping agents to additional info dictionaries.
        _agent_selector (agent_selector): A PettingZoo utility to manage turn order among agents.
        agent_selection (str): The currently selected agent.
        char_encoding (dict): Encoding of the board characters for observation space.
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "name": "sternhalma_v0",
        "is_parallelizable": False,
        "render_fps": 2,
    }

    def __init__(self, num_players: int, board_diagonal: int, render_mode: Optional[str]):
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
        self.possible_agents = self.agents[:]

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.char_encoding = {' ': -2, 'O': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, '|': -1}

        self.observation_spaces = {agent: spaces.Box(low=-2, high=6, shape=(h - 1, w - 1), dtype=np.int8) for agent in
                                   self.agents}

        # Jumping over all the pieces
        max_length = num_players * ((board_diagonal // 2) * (board_diagonal // 2 + 1)) // 2

        self.action_spaces = {agent: VariableLengthTupleSpace(max_length=max_length, low=0, high=max(h - 1, w - 1)) for
                              agent in self.agents}

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.screen = None

        if self.render_mode in ["human", "rgb_array"]:
            self.hex_diagonal = 20
            self.hex_height = math.sqrt(3) * self.hex_diagonal / 2

            self.piece_radius = 10

            height = int(self.hex_height * 3 * board_diagonal)
            width = board_diagonal * self.hex_diagonal * 3

            self.window_size = (width, height)

            self.clock = pygame.time.Clock()

            self.piece_colors = [(128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 80, 0), (75, 0, 130)]
            self.home_colors = [(166, 0, 0), (0, 166, 0), (0, 0, 166), (166, 166, 0), (166, 104, 0), (97, 0, 169)]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> None:
        self.agents = self.possible_agents[:]

        self.board.initialize_board()

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self._clear_rewards()
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.infos[self.agent_selection]['valid_moves'] = self.get_available_actions(self.agent_selection)

        if self.render_mode == "human":
            self.render()

    def step(self, action: List[Tuple[int, int]]) -> None:
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._was_dead_step(action)

        agent = self.agent_selection
        player_idx = self.agents.index(agent)
        self._clear_rewards()

        move = self.convert_action_to_move(action)
        if self.board.is_valid_move(move, player_idx):
            self.board.make_move(player_idx, move)
            reward = self.calculate_reward(player_idx, move)
            if self.check_termination(player_idx):
                self.terminations = {name: True for name in self.agents}
                self.rewards = {name: -10 for name in self.agents}
                self.rewards[agent] = 10
        else:
            # For invalid moves
            reward = -1.0  # Penalty for invalid move
            info = {'invalid_move': True, 'valid_moves': self.infos[agent]['valid_moves']}
            self.rewards[agent] = reward
            self.infos[agent] = info
            self._accumulate_rewards()
            return

        self.rewards[agent] += reward
        self.infos[agent] = {}

        self._accumulate_rewards()

        # Move to the next agent
        self.agent_selection = self._agent_selector.next()
        next_agent = self.agent_selection
        self.infos[next_agent] = {'valid_moves': self.get_available_actions(next_agent)}

        if self.render_mode == "human":
            self.render()

    def observe(self, agent: str) -> np.ndarray:
        """
        Returns the observation for a given agent.

        Args:
            agent (str): The name of the agent to observe the environment for.

        Returns:
            np.ndarray: The observation of the environment for the specified agent.
        """
        return self.state()

    def state(self) -> np.ndarray:
        board_array = np.array(self.board.get_grid())
        observation = np.vectorize(self.char_encoding.get)(board_array[1:, 1:])
        return observation

    def convert_action_to_move(self, action: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Converts an action received from an agent to the corresponding move on the board.

        Args:
            action (Dict[str, Any]): The action to convert.
            agent (str): The name of the agent performing the action.

        Returns:
            List[Tuple[int, int]]: The corresponding move on the board.
        """
        return [(cell[0] + 1, cell[1] + 1) for cell in action]

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
        player_idx = self.agents.index(agent)
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
        start_position = move[0]
        final_position = move[-1]
        if self.board.is_in_home_triangle(final_position, player_idx) and not self.board.is_in_home_triangle(
                start_position, player_idx):
            return 1.0
        else:
            return 0.0

    def check_termination(self, player_idx: int) -> bool:
        """
        Check termination after an action is taken.

        Args:
            player_idx (int): The index of the player who took the action.

        Returns:
            bool: A boolean indicating the termination for the player who took the action.
        """

        return self.board.check_winner(player_idx)

    def render(self):
        """
        Renders the current state of the environment.

        Raises:
            NotImplementedError: If an unsupported render mode is specified.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
        elif self.render_mode == "ansi":
            return str(self.board)
        elif self.render_mode in {"human", "rgb_array"}:
            return self._render_gui()
        else:
            raise ValueError(
                f"{self.render_mode} is not a valid render mode. Available modes are: {self.metadata['render_modes']}"
            )

    def _render_gui(self):
        if self.screen is None:
            pygame.init()

            if self.render_mode == "human":
                pygame.display.set_caption("Sternhalma")
                self.screen = pygame.display.set_mode(self.window_size)
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface(self.window_size)

        self.screen.fill((255, 255, 255))
        board = self.board.get_grid()

        for row_index, row in enumerate(board[1:], start=1):  # Skip the first row of labels
            y = row_index * self.hex_height
            for col_index, cell in enumerate(row[1:], start=1):  # Skip the first column of labels
                x = col_index * 3 * self.hex_diagonal / 2
                if cell != ' ' and cell != '|':
                    self._draw_hexagon(self.screen, (x, y), (255, 255, 255), (0, 0, 0))
                    for i, _ in enumerate(self.agents):
                        home = self.board.get_home(i)
                        if (row_index, col_index) in home:
                            self._draw_hexagon(self.screen, (x, y), self.home_colors[i], (0, 0, 0))

        for i, _ in enumerate(self.agents):
            pieces = self.board.get_player_pieces(i)
            for piece in pieces:
                x = piece[1] * 3 * self.hex_diagonal / 2
                y = piece[0] * self.hex_height
                self._draw_circle(self.screen, (x, y), self.piece_colors[i])

        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def _draw_hexagon(self, surface, center, color, perimeter_color):
        points = []
        for i in range(6):
            angle_deg = 60 * i
            angle_rad = math.pi / 180 * angle_deg
            points.append((center[0] + self.hex_diagonal * math.cos(angle_rad),
                           center[1] + self.hex_diagonal * math.sin(angle_rad)))
        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, perimeter_color, points, 1)

    def _draw_circle(self, surface, center, fill_color):
        # Draw the filled circle
        pygame.draw.circle(surface, fill_color, center, self.piece_radius)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
