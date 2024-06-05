import heapq

class PuzzleState:
    def __init__(self, board, moves=0):
        self.board = board
        self.moves = moves
        self.identifier = tuple(map(tuple, board))

    def __str__(self):
        return "\n".join(["\t".join(map(str, row)) for row in self.board])

    def __lt__(self, other):
        return self.moves < other.moves

def get_blank_position(board):
    for i in range(3):
        for j in range(3):
            if board[i][j] == "*":
                return i, j

def is_goal_state(board):
    goal_state = [[7, 8, 1], [6, "*", 2], [5, 4, 3]]
    return board == goal_state

def is_valid_move(i, j):
    return 0 <= i < 3 and 0 <= j < 3

def apply_move(board, move):
    i, j = get_blank_position(board)
    new_board = [row.copy() for row in board]
    new_i, new_j = i + move[0], j + move[1]

    if is_valid_move(new_i, new_j):
        new_board[i][j], new_board[new_i][new_j] = new_board[new_i][new_j], new_board[i][j]
        return new_board
    else:
        return None

def get_possible_moves(board):
    i, j = get_blank_position(board)
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    valid_moves = [(di, dj) for di, dj in moves if is_valid_move(i + di, j + dj)]
    return valid_moves

def misplaced_tiles(board):
    goal_state = [[7, 8, 1], [6, "*", 2], [5, 4, 3]]
    misplaced_count = 0

    for i in range(3):
        for j in range(3):
            if board[i][j] != goal_state[i][j]:
                misplaced_count += 1

    return misplaced_count

def iterative_deepening(initial_state, max_depth):
    for depth_limit in range(max_depth + 1):
        solution_path, num_moves, num_states_enqueued = dfs(initial_state, depth_limit)
        if num_moves != -1:
            return solution_path, num_moves, num_states_enqueued

    return [], -1, -1

def manhattan_distance(board):
    goal_state = [[7, 8, 1], [6, "*", 2], [5, 4, 3]]
    distance = 0

    for i in range(3):
        for j in range(3):
            if board[i][j] != "*":
                value = board[i][j]
                for row in range(3):
                    if value in goal_state[row]:
                        target_i, target_j = row, goal_state[row].index(value)
                        distance += abs(i - target_i) + abs(j - target_j)

    return distance

def a_star(initial_state, heuristic, max_depth):
    heap = [(heuristic(initial_state.board) + 0, initial_state.moves, initial_state)]
    visited = set()
    solution_path = []

    while heap:
        _, _, current_state = heapq.heappop(heap)

        if current_state.identifier in visited:
            continue

        solution_path.append(current_state)

        if is_goal_state(current_state.board):
            return solution_path, len(solution_path) - 1, len(visited)

        visited.add(current_state.identifier)

        possible_moves = get_possible_moves(current_state.board)
        for move in possible_moves:
            new_board = apply_move(current_state.board, move)
            new_state = PuzzleState(new_board, current_state.moves + 1)
            if new_state.moves <= max_depth:
                heapq.heappush(heap, (heuristic(new_board) + new_state.moves, new_state.moves, new_state))

    return solution_path, -1, len(visited)

def dfs(initial_state, max_depth):
    stack = [(initial_state, 0)]
    visited = set()
    solution_path = []

    while stack:
        current_state, depth = stack.pop()
        if current_state.identifier in visited or depth > max_depth:
            continue

        solution_path.append(current_state)

        if is_goal_state(current_state.board):
            return solution_path, len(solution_path) - 1, len(visited)

        visited.add(current_state.identifier)

        possible_moves = get_possible_moves(current_state.board)
        for move in possible_moves:
            new_board = apply_move(current_state.board, move)
            if new_board is not None:
                new_state = PuzzleState(new_board, current_state.moves + 1)
                stack.append((new_state, depth + 1))

    return solution_path, -1, len(visited)

def print_solution(sequence, num_moves, num_states_enqueued):
    if not sequence:
        print("Goal not reached within the specified depth limit.")
        return

    for state in sequence:
        print(state)
        print()

    print(f"Number of moves = {num_moves}")
    print(f"Number of states enqueued = {num_states_enqueued}")

if __name__ == "__main__":
    initial_board = [[6, 7, 1], [8, 2, "*"], [5, 4, 3]]
    initial_state = PuzzleState(initial_board)

    print("\nInitial state:")
    print(initial_state)

    # DFS
    max_depth_dfs = 10
    solution_sequence_dfs, num_moves_dfs, num_states_enqueued_dfs = dfs(initial_state, max_depth_dfs)

    print("\nDFS Solution:")
    print_solution(solution_sequence_dfs, num_moves_dfs, num_states_enqueued_dfs)

    # Iterative Deepening Search
    max_depth_ids = 10
    solution_sequence_ids, num_moves_ids, num_states_enqueued_ids = iterative_deepening(initial_state, max_depth_ids)

    print("\nIDS Solution:")
    print_solution(solution_sequence_ids, num_moves_ids, num_states_enqueued_ids)

    # A* using Manhattan distance heuristic
    max_depth_astar_manhattan = 10
    solution_sequence_astar_manhattan, num_moves_astar_manhattan, num_states_enqueued_astar_manhattan = a_star(initial_state, manhattan_distance, max_depth_astar_manhattan)
    print("\nA* using Manhattan distance Solution:")
    print_solution(solution_sequence_astar_manhattan, num_moves_astar_manhattan, num_states_enqueued_astar_manhattan)

    # A* using misplaced tiles heuristic
    max_depth_astar_tiles = 10
    solution_sequence_astar_tiles, num_moves_astar_tiles, num_states_enqueued_astar_tiles = a_star(initial_state, misplaced_tiles, max_depth_astar_tiles)
    print("\nA* using misplaced tiles Solution:")
    print_solution(solution_sequence_astar_tiles, num_moves_astar_tiles, num_states_enqueued_astar_tiles)
