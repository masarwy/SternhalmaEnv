from typing import List, Tuple

class Player:
    def __init__(self, piece_symbol: str):
        self.piece_symbol = piece_symbol
        self.pieces: List[Tuple[int, int]] = []
        self.home_triangle: int

    def set_pieces(self, pieces: List[Tuple[int, int]]) -> None:
        self.pieces = pieces

    def update_pieces(self, src: Tuple[int, int], dst: Tuple[int, int]) -> None:
        self.pieces = [dst if item == src else item for item in self.pieces]

    def get_pieces(self) -> List[Tuple[int, int]]:
        return self.pieces

    def set_home_triangle(self, triangle: int) -> None:
        self.home_triangle = triangle

    def get_home_triangle(self) -> int:
        return self.home_triangle

    def get_piece(self) -> str:
        return self.piece_symbol
