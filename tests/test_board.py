import unittest

from sternhalma.utils.board import Board


class BoardTests(unittest.TestCase):
    def test_initializes_expected_piece_count_for_two_players(self):
        board = Board(diagonal=5, num_players=2)
        counts = [len(p.get_pieces()) for p in board.players]
        self.assertEqual(counts, [3, 3])

    def test_available_move_is_valid_and_updates_piece_position(self):
        board = Board(diagonal=5, num_players=2)
        move = board.get_available_moves(0)[0]
        start, end = move[0], move[-1]

        self.assertTrue(board.is_valid_move(move, 0))
        self.assertTrue(board.make_move(0, move))
        self.assertEqual(board.grid[start[0]][start[1]], "O")
        self.assertEqual(board.grid[end[0]][end[1]], "A")

    def test_rejects_illegal_multihop_sequence(self):
        board = Board(diagonal=5, num_players=2)
        illegal_move = [(6, 8), (13, 9), (6, 9), (8, 9), (5, 1)]
        self.assertFalse(board.is_valid_move(illegal_move, 0))
        self.assertFalse(board.make_move(0, illegal_move))


if __name__ == "__main__":
    unittest.main()
