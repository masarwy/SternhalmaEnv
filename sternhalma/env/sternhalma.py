import math
import numpy as np
import gymnasium
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from typing import Dict, Any, List, Tuple, Optional, TypedDict
from gymnasium import spaces
import pygame

from sternhalma.utils.board import Board
from sternhalma.utils.types import VariableLengthTupleSpace, HandleNoOpWrapper

# Terminal reward magnitude for win/loss. Kept separate from per-move shaping
# rewards so that downstream reward-scaling wrappers do not accidentally crush
# the win/loss signal.
_WIN_REWARD: float = 10.0
_LOSS_REWARD: float = -10.0


class Metadata(TypedDict):
    render_modes: List[str]
    name: str
    is_parallelizable: bool
    render_fps: int


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = HandleNoOpWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    """
    A PettingZoo environment for the Sternhalma game (also known as Chinese Checkers).

    The environment supports multiple rendering modes and manages the game state, including the board,
    player pieces, and actions. It adheres to the PettingZoo AECEnv interface, providing methods for stepping
    through the game, observing the state, and rendering.
    """

    metadata: Metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "name": "sternhalma_v0",
        "is_parallelizable": False,
        "render_fps": 2,
    }

    VALID_REWARD_MODES = {"sparse", "dense", "potential_shaped"}

    # Win/loss magnitude (accessible on instances for wrappers that need it).
    WIN_REWARD: float = _WIN_REWARD
    LOSS_REWARD: float = _LOSS_REWARD

    def __init__(
            self,
            num_players: int,
            board_diagonal: int,
            render_mode: Optional[str],
            reward_mode: str = "sparse",
            gamma: float = 1.0,
    ):
        """
        Initializes the Sternhalma environment with the specified number of players and board size.

        Args:
            num_players (int): Number of players in the game.
            board_diagonal (int): The size of the game board, measured diagonally across.
            render_mode (Optional[str]): The mode used for rendering. Can be 'human', 'ansi', or 'rgb_array'.
            reward_mode (str): Reward calculation mode. One of: "sparse", "dense", "potential_shaped".
            gamma (float): Discount factor for the potential-shaping term in "potential_shaped" mode.
                This should match the discount factor (gamma) used by the RL algorithm so that the
                shaped reward F(s,s') = gamma*phi(s') - phi(s) preserves the optimal policy
                (Ng et al., 1999; Lu et al., 2014).

        Raises:
            ValueError: If the board diagonal is not odd or less than 3, or if an invalid number of players is specified.
        """
        super().__init__()

        if board_diagonal < 3 or board_diagonal % 2 == 0:
            raise ValueError("board_diagonal must be an odd number and greater than or equal to 3.")

        if num_players not in [2, 3, 4, 6]:
            raise ValueError("num_players must be one of the following values: 2, 3, 4, 6.")

        if reward_mode not in self.VALID_REWARD_MODES:
            raise ValueError(
                f"reward_mode must be one of {sorted(self.VALID_REWARD_MODES)}, got: {reward_mode!r}."
            )
        if not isinstance(gamma, (int, float)) or not 0.0 <= float(gamma) <= 1.0:
            raise ValueError(f"gamma must be a float in [0, 1], got: {gamma!r}.")

        self.num_players = num_players
        self.reward_mode = reward_mode
        self.gamma = float(gamma)
        self.board = Board(board_diagonal, num_players)
        h, w = self.board.get_dims()

        self.agents = [f"player_{i}" for i in range(self.num_players)]
        self.possible_agents = self.agents[:]

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {agent: 0. for agent in self.agents}
        self._cumulative_rewards = {name: 0. for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.infos: Dict[str, Dict] = {agent: {} for agent in self.agents}

        self.char_encoding = {' ': -2, 'O': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, '|': -1}

        # Number of per-piece distance features: one scalar per piece per player.
        # For player p with k pieces, we include min hex-distance of each piece to
        # player p's home triangle, normalised by the board's maximum possible distance.
        pieces_per_player = (board_diagonal // 2) * (board_diagonal // 2 + 1) // 2
        self._max_board_dist = float(3 * board_diagonal)  # loose upper bound for normalisation
        # distance features: num_players * pieces_per_player values in [0, 1]
        num_dist_features = self.num_players * pieces_per_player

        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "board": spaces.Box(low=-2, high=6, shape=(h - 1, w - 1), dtype=np.int8),
                    "current_player": spaces.Discrete(self.num_players),
                    # Per-piece hex distances to *each player's own* home triangle,
                    # normalised to [0, 1].  Shape: (num_players * pieces_per_player,)
                    "distances_to_home": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(num_dist_features,),
                        dtype=np.float32,
                    ),
                }
            )
            for agent in self.agents
        }

        # Jumping over all the pieces
        max_length = num_players * ((board_diagonal // 2) * (board_diagonal // 2 + 1)) // 2

        self.action_spaces = {
            agent: VariableLengthTupleSpace(
                max_length=max_length,
                low=0,
                high=max(h - 1, w - 1),
                allow_noop=True,
            )
            for agent in self.agents
        }

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
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.infos[self.agent_selection]['valid_moves'] = self.get_available_actions(self.agent_selection)

        if self.render_mode == "human":
            self.render()

    def step(self, action: Optional[List[Tuple[int, int]]]) -> None:
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._was_dead_step(action)

        agent = self.agent_selection
        player_idx = self.agents.index(agent)
        self._clear_rewards()

        normalized_action = self.normalize_action(action)
        if normalized_action == tuple():
            self._accumulate_rewards()
            self.skip_turn()
            return

        if not self.infos[agent]['valid_moves']:  # If no available actions, skip to next player
            self._accumulate_rewards()  # Accumulate rewards if any before skipping
            self.agent_selection = self._agent_selector.next()  # Move to the next agent
            self.infos[self.agent_selection] = {'valid_moves': self.get_available_actions(self.agent_selection)}
            return

        valid_actions = {self.normalize_action(candidate) for candidate in self.infos[agent]['valid_moves']}
        if normalized_action is None or normalized_action not in valid_actions:
            reward = -1.0
            info = {'invalid_move': True, 'valid_moves': self.infos[agent]['valid_moves']}
            self.rewards[agent] = reward
            self.infos[agent] = info
            self._accumulate_rewards()
            return

        move = self.convert_action_to_move(list(normalized_action))
        if self.board.is_valid_move(move, player_idx):
            self.board.make_move(player_idx, move)
            reward = self.calculate_reward(player_idx, move, self.gamma)
            if self.check_termination(player_idx):
                # ---- FIX: terminal rewards are set BEFORE _accumulate_rewards so
                # that ALL agents (including losers) see their terminal signal in
                # the same step.  Losing agents are then removed via _was_dead_step
                # on the next call, but their terminal reward is already accumulated.
                self.terminations = {name: True for name in self.agents}
                # Assign shaped move reward to winner first, then overwrite with
                # the win bonus.  Losers receive the flat loss penalty; the shaped
                # reward for the winning move is intentionally NOT added to the
                # loser's signal to avoid sign confusion.
                self.rewards = {name: self.LOSS_REWARD for name in self.agents}
                self.rewards[agent] = self.WIN_REWARD
                # Do NOT add the per-move shaping reward on top of the terminal
                # signal — the terminal reward dominates and mixing them conflates
                # two separate learning signals.
        else:
            # For invalid moves
            reward = -1.0  # Penalty for invalid move
            info = {'invalid_move': True, 'valid_moves': self.infos[agent]['valid_moves']}
            self.rewards[agent] = reward
            self.infos[agent] = info
            self._accumulate_rewards()
            return

        if not self.terminations[agent]:
            # Only accumulate the shaping reward on non-terminal steps.
            self.rewards[agent] += reward
        self.infos[agent] = {}

        self._accumulate_rewards()

        # Move to the next agent
        self.agent_selection = self._agent_selector.next()
        next_agent = self.agent_selection
        self.infos[next_agent] = {'valid_moves': self.get_available_actions(next_agent)}

        if self.render_mode == "human":
            self.render()

    def skip_turn(self) -> None:
        """Advances to the next agent, effectively skipping the current agent's turn."""
        self.agent_selection = self._agent_selector.next()
        self.infos[self.agent_selection] = {'valid_moves': self.get_available_actions(self.agent_selection)}

    def observe(self, agent: str) -> Dict[str, Any]:
        """
        Returns the observation for a given agent.

        Args:
            agent (str): The name of the agent to observe the environment for.

        Returns:
            dict: Observation containing:
                - "board": encoded board matrix (int8).
                - "current_player": index of the agent whose turn it is.
                - "distances_to_home": per-piece hex distances to each player's own
                  home triangle, normalised to [0, 1].  This gives the policy a
                  compact spatial signal without requiring it to re-learn hex geometry
                  from the raw board encoding.
        """
        # Fix 1: guard against dead-agent observation calls (e.g. after termination
        # at diagonal>=7 the wrapper stack may still call observe() before the
        # agent is removed from env.agents).
        if self.agent_selection in self.agents:
            current_player_idx = int(self.agents.index(self.agent_selection))
        else:
            current_player_idx = 0

        # Fix 2: current_player reports whose turn it is (agent_selection),
        # which is the correct Markov signal for all agents including the
        # observer — it tells the policy whether it is acting or waiting.
        return {
            "board": self.state(),
            "current_player": current_player_idx,
            "distances_to_home": self._compute_distances_to_home(),
        }

    def _compute_distances_to_home(self) -> np.ndarray:
        """
        Compute per-piece hex distances to each player's own home triangle.

        Returns a flat float32 array of shape (num_players * pieces_per_player,)
        where each entry is the hex distance of a piece to the nearest cell in that
        player's home triangle, normalised by ``_max_board_dist``.  Missing pieces
        (already in home or captured) are represented by 0.0.

        The ordering is: [player_0_piece_0, ..., player_0_piece_k,
                          player_1_piece_0, ..., player_1_piece_k, ...]
        """
        pieces_per_player = (
            self.board.diagonal // 2 * (self.board.diagonal // 2 + 1) // 2
        )
        features: List[float] = []
        for player_idx in range(self.num_players):
            pieces = list(self.board.get_player_pieces(player_idx))
            # Pad / truncate to a fixed length for a stable observation shape.
            for i in range(pieces_per_player):
                if i < len(pieces):
                    dist = float(self._distance_to_home(pieces[i], player_idx))
                    features.append(min(dist / self._max_board_dist, 1.0))
                else:
                    features.append(0.0)
        return np.array(features, dtype=np.float32)

    def state(self) -> np.ndarray:
        board_array = np.array(self.board.get_grid())
        observation = np.vectorize(self.char_encoding.get)(board_array[1:, 1:]).astype(np.int8)
        return observation

    def convert_action_to_move(self, action: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Converts an action received from an agent to the corresponding move on the board.

        Args:
            action (Dict[str, Any]): The action to convert.

        Returns:
            List[Tuple[int, int]]: The corresponding move on the board.
        """
        return [(cell[0] + 1, cell[1] + 1) for cell in action]

    def normalize_action(self, action: Any) -> Optional[Tuple[Tuple[int, int], ...]]:
        """
        Convert an action-like input into a hashable canonical tuple-of-tuples.
        Returns None when the shape or element types are invalid.
        """
        if action is None:
            return tuple()

        if not isinstance(action, list):
            return None

        normalized: List[Tuple[int, int]] = []
        for cell in action:
            if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                return None
            if not isinstance(cell[0], (int, np.integer)) or not isinstance(cell[1], (int, np.integer)):
                return None
            normalized.append((int(cell[0]), int(cell[1])))
        return tuple(normalized)

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

    @staticmethod
    def _to_axial(position: Tuple[int, int]) -> Tuple[float, float]:
        # Board coordinates are represented on a skewed 2D grid; convert to axial for hex-distance math.
        row, col = position
        return float(col), float((row - col) / 2.0)

    @classmethod
    def _hex_distance(cls, source: Tuple[int, int], target: Tuple[int, int]) -> int:
        source_q, source_r = cls._to_axial(source)
        target_q, target_r = cls._to_axial(target)
        dq = source_q - target_q
        dr = source_r - target_r
        return int(round((abs(dq) + abs(dr) + abs(dq + dr)) / 2.0))

    def _distance_to_home(self, position: Tuple[int, int], player_idx: int) -> int:
        home_positions = self.board.get_home(player_idx)
        if not home_positions:
            return 0
        return min(self._hex_distance(position, home_position) for home_position in home_positions)

    def _phi(self, position, player_idx):
        return -self._distance_to_home(position, player_idx)

    def sparse_reward(self, player_idx: int, move: List[Tuple[int, int]]) -> float:
        start_position = move[0]
        final_position = move[-1]
        if self.board.is_in_home_triangle(final_position, player_idx) and not self.board.is_in_home_triangle(
                start_position, player_idx
        ):
            return 1.0
        return 0.0

    def dense_reward(self, player_idx: int, move: List[Tuple[int, int]]) -> float:
        start_position = move[0]
        final_position = move[-1]
        start_distance = self._distance_to_home(start_position, player_idx)
        final_distance = self._distance_to_home(final_position, player_idx)
        return float(start_distance - final_distance)

    def potential_shaped_reward(self, player_idx: int, move: List[Tuple[int, int]], gamma: float) -> float:
        """Potential-based shaping: F(s, s') = gamma * phi(s') - phi(s).

        The shaping term preserves Nash equilibria in general-sum stochastic
        games when gamma here matches the RL algorithm's discount factor
        (Lu et al., 2014; Ng et al., 1999).  Pass ``env.gamma`` (which should
        equal the algorithm's gamma) when calling this method.

        Args:
            player_idx: index of the acting player.
            move: list of board positions traversed by the move.
            gamma: discount factor — MUST match the RL algorithm's gamma.

        Returns:
            Scalar shaped reward = sparse_reward + F(s, s').
        """
        start = move[0]
        final = move[-1]

        base_reward = self.sparse_reward(player_idx, move)

        # F(s, s') = gamma * phi(s') - phi(s)
        shaping = gamma * self._phi(final, player_idx) - self._phi(start, player_idx)

        return base_reward + shaping

    def calculate_reward(self, player_idx: int, move: List[Tuple[int, int]], gamma: float = 1.0) -> float:
        """
        Calculate the reward for the acting agent according to `self.reward_mode`.

        Args:
            player_idx (int): The index of the player who made the move.
            move (List[Tuple[int, int]]): The move made by the player.
            gamma (float): Discount factor used only by the "potential_shaped" reward mode.

        Returns:
            float: The reward resulting from the move.
        """
        if self.reward_mode == "sparse":
            return self.sparse_reward(player_idx, move)
        if self.reward_mode == "dense":
            return self.dense_reward(player_idx, move)
        if self.reward_mode == "potential_shaped":
            return self.potential_shaped_reward(player_idx, move, gamma)
        raise RuntimeError(f"Unsupported reward_mode: {self.reward_mode!r}")

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
