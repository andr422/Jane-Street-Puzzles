import random
import numpy as np

# soccer ball modeled as a graph with white panels
# as vertices with edges connecting to adjacent white panels
ball_graph = {
    # https://en.wikipedia.org/wiki/File:Truncated_icosahedron_stereographic_projection_pentagon.png
    # Lettered clockwise, starting at top for each layer
    'A': ['B','F','E'],
    'B': ['C','G','A'],
    'C': ['D','H','B'],
    'D': ['E','I','C'],
    'E': ['A','J','D'],
    'F': ['A','K','O'],
    'G': ['B','K','L'],
    'H': ['C','L','M'],
    'I': ['D','M','N'],
    'J': ['E','N','O'],
    'K': ['F','G','P'],
    'L': ['G','H','Q'],
    'M': ['H','I','R'],
    'N': ['I','J','S'],
    'O': ['J','F','T'],
    'P': ['K','T','Q'],
    'Q': ['L','R','P'],
    'R': ['M','S','Q'],
    'S': ['N','T','R'],
    'T': ['O','P','S']
}


def ball_graph_to_transition_matrix(graph: dict) -> np.ndarray:
    matrix = []
    for key in graph:
        vector = [0 for _ in range(len(graph))]
        for panel_name in graph.get(key):
            vector[ord(panel_name) - 65] = 1/3
        matrix.append(vector)
    return np.array(matrix)


def get_ball_stationary_distribution(transition_matrix: np.ndarray, pi: np.ndarray) -> np.ndarray:
    pi_prime = np.matmul(transition_matrix, pi)
    if np.allclose(pi, pi_prime, 1e-10, 1e-10):
        return pi_prime
    else:
        return get_ball_stationary_distribution(transition_matrix, pi_prime)


def solve_expected_steps_on_ball() -> int:
    soccer_ball_transition_matrix = ball_graph_to_transition_matrix(ball_graph)
    # ant starts at state 0
    pi = np.array([1 if i == 0 else 0 for i in range(len(ball_graph))])
    stationary_distribution = get_ball_stationary_distribution(soccer_ball_transition_matrix, pi)
    return round(1/stationary_distribution[0])


def simulate_ball_walk(number_of_trials: int, starting_position='A') -> float:
    sum_of_moves = 0
    for i in range(number_of_trials):
        current_position = random.choice(ball_graph.get(starting_position))
        moves = 1
        while current_position != starting_position:
            current_position = random.choice(ball_graph.get(current_position))
            moves += 1
        sum_of_moves += moves
    return sum_of_moves/number_of_trials


"""
I represent the tiling as a cartesian coordinate system stored as a graph with each white tile represented
as a point and connected to three other white tiles. I observed white tiles appear vertically 
stacked in groups of two separated by black tiles. Therefore, "top" tiles can only move down a tile
and "bottom" tiles can only move up a tile. However, each white tile can access another white
tile in both horizontal direction, totalling 3 accessible tiles. I also noticed tiles alternate
vertical position (top/bottom) in all four directions. I represented the origin (0,0) as a "bottom tile"
so all white tiles whose x and y coordinates sum to a number n such that n % 2 == 0 are bottom tiles and
 all other white tiles are "top tiles"
"""


def make_tile_graph(expected_steps: int) -> dict:
    # once ant reaches edge,
    size = expected_steps // 2 + 2
    graph = {}
    for i in range(size):
        for j in range(size):
            # once ant reaches edge, reaching origin before expected value of walk on ball is impossible
            if i + j <= size:
                if (i+j)%2 == 0:
                    graph.update({(i,j): [(i+1, j), (i, j+1), (i, j-1)],
                                  (-i,j): [(-i+1, j), (-i, j+1), (-i, j-1)],
                                  (-i,-j): [(-i+1, -j), (-i, -j+1), (-i, -j-1)],
                                  (i,-j): [(i+1, -j), (i, -j+1), (i, -j-1)]})
                else:
                    graph.update({(i,j): [(i-1, j), (i, j+1), (i, j-1)],
                                  (-i,j): [(-i-1, j), (-i, j+1), (-i, j-1)],
                                  (-i,-j): [(-i-1, -j), (-i, -j+1), (-i, -j-1)],
                                  (i,-j): [(i-1, -j), (i, -j+1), (i, -j-1)]})
    return graph


def tile_graph_to_transition_matrix(graph: dict) -> np.ndarray:
    matrix = []
    keys = list(graph)
    number_of_states = len(graph)
    for location in graph:
        vector = [0 for _ in range(number_of_states)]
        for new_state in graph.get(location):
            if new_state in keys:
                vector[keys.index(new_state)] = 1/3
        matrix.append(vector)
    return np.array(matrix)


def get_answer(expected_steps) -> float:
    graph = make_tile_graph(expected_steps)
    matrix = tile_graph_to_transition_matrix(graph)
    locations = list(graph)
    pi = [1 if index == 0 else 0 for index in range(len(locations))]
    pi = np.matmul(matrix, pi)
    moves = 1
    over = 0
    under = 0
    while not np.isclose(1,over+under,1e-6, 1e-6):
        # Random walk ends after returning to origin -- track whether
        # above or below expected value of random walk on ball
        if moves <= expected_steps:
            under += pi[0]
            pi[0] = 0.0
        else:
            over += sum(pi)
            pi = [0 for _ in pi]

        # tracks probability ant is too far to return to origin before expected value
        for index in range(len(pi)):
            if (abs(locations[index][0])+abs(locations[index][1])) > (expected_steps-moves):
                over += pi[index]
                pi[index] = 0.0

        pi = np.matmul(matrix,pi)
        moves += 1

    return over


def get_moves(x: tuple) -> list:
    if (abs(x[0]) + abs(x[1])) % 2 == 0:
        return [(1, 0), (0, 1), (0, -1)]
    else:
        return [(-1, 0), (0, 1), (0, -1)]


def simulate_tile_walk(number_of_trials: int, expected_steps: int) -> float:
    graph = make_tile_graph(expected_steps)
    over_limit_counter = 0
    for i in range(number_of_trials):
        current_position = random.choice(graph.get((0,0)))
        moves = 1
        while current_position != (0,0):
            if abs(current_position[0])+abs(current_position[1]) > expected_steps-moves:
                over_limit_counter += 1
                break
            current_position = random.choice(graph.get(current_position))
            moves += 1
    return over_limit_counter/number_of_trials


print(get_answer(solve_expected_steps_on_ball()))
