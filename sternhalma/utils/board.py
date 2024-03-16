from typing import List, Tuple

from grid import generate_board, get_triangle_indices, print_grid
from player import Player


class Board:
    def __init__(self, diagonal: int, num_players: int):
        self.diagonal = diagonal
        self.num_players = num_players
        self.turn = 0
        self.grid = None
        self.width = None
        self.height = None
        self.players = None
        self.initialize_board()

    def get_dims(self):
        return self.height, self.width

    def print_board(self):
        print_grid(self.grid)

    def get_grid(self) -> List[List[str]]:
        return self.grid

    def initialize_board(self):
        self.grid = generate_board(self.diagonal)

        self.height = len(self.grid)
        self.width = len(self.grid[0])

        # Mapping of player numbers to triangle indices
        player_positions = {
            2: [3, 6],
            3: [1, 3, 5],
            4: [1, 2, 4, 5],
            6: [1, 2, 3, 4, 5, 6]
        }
        player_home_triangles = {
            2: [6, 3],
            3: [4, 6, 2],
            4: [4, 5, 1, 2],
            6: [4, 5, 6, 1, 2, 3]
        }

        pieces = ['A', 'B', 'C', 'D', 'E', 'F']
        self.players = [Player(pieces[i]) for i in range(self.num_players)]

        # Get the triangle indices for the current number of players
        positions = player_positions.get(self.num_players, [])
        player_home_triangles = player_home_triangles.get(self.num_players, [])

        # Set the pieces for each player
        for i, player in enumerate(self.players):
            if i < len(positions):  # Check to prevent index out of range
                triangle_index = positions[i]
                pieces = get_triangle_indices(self.grid, self.diagonal, triangle_index)
                player.set_pieces(pieces)
                player.set_home_triangle(player_home_triangles[i])
                for p in pieces:
                    self.grid[p[0]][p[1]] = player.get_piece()

    def is_valid_move(self, positions: List[Tuple[int, int]], player_idx: int) -> bool:
        if len(positions) < 2:
            return False

        if self.grid[positions[0][0]][positions[0][1]] != self.players[player_idx].get_piece():
            return False

        if not (0 <= positions[0][0] < self.height and 0 <= positions[0][1] < self.width):
            return False

        elif len(positions) == 2:
            from_row, from_col = positions[0]
            to_row, to_col = positions[1]

            # Check if to_position is within board bounds
            if not (0 <= to_row < self.height and 0 <= to_col < self.width):
                return False

            if abs(from_row - to_row) == 1 and abs(from_col - to_col) == 1:  # Diagonally adjacent
                return self.grid[to_row][to_col] == 'O'
            elif from_col == to_col and abs(from_row - to_row) == 2:  # Vertically adjacent
                return self.grid[to_row][to_col] == 'O'
            elif abs(from_row - to_row) == 4 and from_col == to_col:  # vertical jump over
                mid_row, mid_col = (to_row + from_row) // 2, (to_col + from_col) // 2
                return self.grid[mid_row][mid_col] != 'O' and self.grid[to_row][to_col] == 'O'
            elif abs(from_row - to_row) == 2 and abs(from_col - to_col) == 2:  # diagonal jump over
                mid_row, mid_col = (to_row + from_row) // 2, (to_col + from_col) // 2
                return self.grid[mid_row][mid_col] != 'O' and self.grid[to_row][to_col] == 'O'
            return False

        else:
            curr_row, curr_col = positions[0]
            for i in range(1, len(positions)):
                next_row, next_col = positions[i]

                if not (0 <= next_row < self.height and 0 <= next_col < self.width):
                    return False

                mid_row, mid_col = (next_row + curr_row) // 2, (next_col + curr_col) // 2
                if self.grid[mid_row][mid_col] == 'O':
                    return False

                # Check if to_position is within board bounds
                if not (0 <= next_row < self.height and 0 <= next_col < self.width):
                    return False

                if abs(curr_row - next_row) == 2 and abs(curr_col - next_col) == 2:
                    if self.grid[next_row][next_col] != 'O':
                        return False

                elif curr_col == next_col and abs(curr_row - next_row) == 4:
                    if self.grid[next_row][next_col] != 'O':
                        return False

                curr_row, curr_col = next_row, next_col

            return True

    def make_move(self, player_idx: int, positions: List[Tuple[int, int]]) -> bool:
        if self.is_valid_move(positions, player_idx):
            # Unpack the positions
            from_row, from_col = positions[0]
            to_row, to_col = positions[-1]

            player = self.players[player_idx]

            # Update the grid to reflect the move
            self.grid[from_row][from_col] = 'O'  # Mark the from_position as available
            self.grid[to_row][to_col] = player.get_piece()  # Set the to_position to player's piece

            player.update_pieces(positions[0], positions[-1])
            self.turn += 1
            return True
        else:
            return False

    def check_winner(self, player_idx: int) -> bool:
        player = self.players[player_idx]
        home = get_triangle_indices(self.grid, self.diagonal, player.get_home_triangle())
        count = 0
        for vertex in home:
            if self.grid[vertex[0]][vertex[1]] == 'O':
                return False
            if self.grid[vertex[0]][vertex[1]] == player.get_piece():
                count += 1
        # Majority Control
        return count > len(home) // 2
