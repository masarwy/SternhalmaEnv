from typing import List, Tuple

def fill_submatrix_and_clear_triangle(matrix: List[List[str]], sub_top_left: Tuple[int, int], sub_length: int,
                                      clear_triangle: str) -> List[List[str]]:
    sub_row, sub_col = sub_top_left  # Top-left corner of the sub-matrix

    temp_matrix = [[' ' for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
    dia = (sub_length + 1) * 2 // 3
    first_char = 'O' if dia % 4 == 1 else '|'
    second_char = '|' if dia % 4 == 1 else 'O'
    # Apply the chessboard pattern within the sub-matrix starting from the top-left corner
    for i in range(sub_row, sub_row + sub_length):
        for j in range(sub_col, sub_col + sub_length):
            if (i + j) % 2 == 0:
                temp_matrix[i][j] = first_char
            else:
                temp_matrix[i][j] = second_char
            if clear_triangle == 'lower-left' and i - sub_row > j - sub_col:
                temp_matrix[i][j] = ' '
            if clear_triangle == 'upper-right' and i - sub_row < j - sub_col:
                temp_matrix[i][j] = ' '
            if clear_triangle == 'upper-left' and i + j - sub_row - sub_col + 1 < sub_length:
                temp_matrix[i][j] = ' '
            if clear_triangle == 'lower-right' and i + j - sub_row - sub_col + 1 > sub_length:
                temp_matrix[i][j] = ' '

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == ' ' and temp_matrix != ' ':
                matrix[i][j] = temp_matrix[i][j]

    return matrix


def generate_board(dia: int) -> List[List[str]]:
    if dia % 2 == 0:
        raise 'Invalid Diagonal'
    width = 2 * dia
    height = 2 + 6 * (dia // 2)
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    for i in range(1, height):
        grid[i][0] = i
    for i in range(1, width):
        grid[0][i] = i

    corner_row = height // 2
    first_corner_col = width // 2 - dia // 2
    edge = dia + dia // 2
    fill_submatrix_and_clear_triangle(grid, (corner_row, first_corner_col), edge, 'lower-right')
    fill_submatrix_and_clear_triangle(grid, (1, 1), edge, 'upper-left')
    fill_submatrix_and_clear_triangle(grid, (1, first_corner_col), edge, 'upper-right')
    fill_submatrix_and_clear_triangle(grid, (corner_row, 1), edge, 'lower-left')
    return grid


def generate_triangle_pattern(matrix_size: int) -> List[List[str]]:
    # Initialize the matrix with spaces
    matrix = [[' ' for _ in range(matrix_size)] for _ in range(matrix_size)]

    # Fill the triangle pattern
    for r in range(matrix_size):
        for c in range((r // 2) + 1):
            if r % 2 == 0:
                # For even rows, place 'O' in the correct columns
                matrix[r][c * 2] = 'O'
            else:
                # For odd rows, place 'O' or '|' based on the column
                if c * 2 + 1 <= r:
                    matrix[r][c * 2 + 1] = '|' if c != r // 2 else 'O'

    return matrix


def print_grid(lists_of_chars: List[List[str]]) -> None:
    rows = len(lists_of_chars)
    cols = len(lists_of_chars[0]) if rows > 0 else 0

    # Find the maximum width of the content in the grid
    max_width = max(len(str(item)) for row in lists_of_chars for item in row)

    # Adjust the cell width based on the maximum content width
    cell_width = max_width + 2  # Adding padding on both sides

    # Print the top border
    print('+' + ('-' * cell_width + '+') * cols)

    for row in lists_of_chars:
        # Print the row with vertical separators and adjusted spacing
        print('|' + '|'.join(f'{char:^{cell_width}}' for char in row) + '|')

        # Print the horizontal separator after the row
        print('+' + ('-' * cell_width + '+') * cols)


def get_triangle_indices(grid: List[List[str]], board_size: int, triangle: int) -> List[Tuple[int, int]]:
    vertices = []
    triangle_edge = board_size // 2
    if triangle == 6:
        mid_row = len(grid) // 2 + 1
        for i in range(triangle_edge + 2):
            start = mid_row + (1 - i)
            end = mid_row + (i - 1)
            for j in range(start, end, 2):
                vertices.append((j, i - 1))
    if triangle == 3:
        mid_row = len(grid) // 2 + 1
        for i in range(triangle_edge + 2):
            start = mid_row + (1 - i)
            end = mid_row + (i - 1)
            for j in range(start, end, 2):
                vertices.append((j, len(grid[0]) - i + 1))
    if triangle == 2:
        triangle_side = board_size // 2
        first_row = 1
        first_col = len(grid[0]) - triangle_side - 1
        for i in range(triangle_side):
            for j in range(triangle_side - i):
                vertices.append((first_row + i + 2 * j, first_col - i))
    if triangle == 1:
        triangle_side = board_size // 2
        first_row = 1
        first_col = triangle_side + 1
        for i in range(triangle_side):
            for j in range(triangle_side - i):
                vertices.append((first_row + i + 2 * j, first_col + i))
    if triangle == 5:
        triangle_side = board_size // 2
        first_row = len(grid) - 1
        first_col = triangle_side + 1
        for i in range(triangle_side):
            for j in range(triangle_side - i):
                vertices.append((first_row - i - 2 * j, first_col + i))
    if triangle == 4:
        triangle_side = board_size // 2
        first_row = len(grid) - 1
        first_col = len(grid[0]) - triangle_side - 1
        for i in range(triangle_side):
            for j in range(triangle_side - i):
                vertices.append((first_row - i - 2 * j, first_col - i))
    return vertices
