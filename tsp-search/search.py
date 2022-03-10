import math

import numpy as np


def bfs(graph, start):
    return _generic_exhaustive_search(graph, start, 0)


def dfs(graph, start):
    return _generic_exhaustive_search(graph, start, -1)


def _generic_exhaustive_search(graph, start, pop_position):
    states = [(0, [start])]
    acceptable_states = []

    while states:
        state = states.pop(pop_position)

        if _is_acceptable_state(graph, state):
            acceptable_states.append(state)
        else:
            next_states = _get_next_states(graph, state)
            states.extend(next_states)

    acceptable_states.sort()
    optimal_state = acceptable_states[0] if acceptable_states else None

    if optimal_state is None:
        return None

    optimal_cost, optimal_path = optimal_state
    return optimal_path, optimal_cost


def nearest_neighbor(graph, start):
    cities_count = graph.shape[0]
    state = (0, [start])

    for i in range(cities_count):
        next_states = _get_next_states(graph, state)

        if not next_states:
            return None

        best_next_state = sorted(next_states)[0]
        state = best_next_state

    cost, path = state
    return path, cost


def nearest_insertion(graph, start):
    cities_count = graph.shape[0]
    path = [start, start]
    # print(graph)
    # print(path, _calc_path_cost(graph, path))

    for i in range(cities_count - 1):

        nearest_city = _get_city_nearest_to_path(graph, path)
        # print('Best city:', nearest_city)

        if not nearest_city:
            return

        path = _insert_city_to_path(graph, path, nearest_city)

        if path is None:
            return

    cost = _calc_path_cost(graph, path)
    return path, cost


def _get_city_nearest_to_path(graph, path):
    cities_count = graph.shape[0]
    not_visited_cities = [i for i in range(cities_count) if i not in path]
    city_distances = []
    for target_city in not_visited_cities:
        minimum_distance = math.inf
        for start_city in path:
            distance = graph[start_city, target_city]
            if distance != -math.inf:
                minimum_distance = min(minimum_distance, distance)
        if minimum_distance != math.inf:
            city_distance = (minimum_distance, target_city)
            city_distances.append(city_distance)
    # print('Nearest cities:', sorted(city_distances))
    nearest_city = sorted(city_distances)[0][1] if city_distances else None
    return nearest_city


def _insert_city_to_path(graph, path, city):
    possible_paths = [path[:pos] + [city] + path[pos:] for pos in range(1, len(path))]
    # print(possible_paths)
    paths_lengths = [(_calc_path_cost(graph, path), path) for path in possible_paths]
    # print('All insertions:', paths_lengths)
    possible_paths_lengths = [path_length for path_length in paths_lengths if path_length and path_length[0] != -math.inf]
    # print('Possible insertions:', possible_paths_lengths)
    shortest_path = sorted(possible_paths_lengths)[0][1] if possible_paths_lengths else None
    return shortest_path


def _calc_path_cost(graph, path):
    total_cost = 0
    for current_city, next_city in zip(path, path[1:]):
        cost = graph[current_city, next_city]
        # print(f'cost {current_city}={next_city}: {cost}')
        total_cost += cost
    return total_cost


def _is_acceptable_state(graph, state):
    cities_count = graph.shape[0]
    path_length = len(state[1])
    return path_length == cities_count + 1


def _get_next_states(graph, state):
    cities_count = graph.shape[1]
    current_path_length = len(state[1])

    if current_path_length == cities_count:
        return _get_next_states_final_step(graph, state)
    else:
        return _get_next_states_standard_step(graph, state)


def _get_next_states_standard_step(graph, state):
    next_states = []
    cities_count = graph.shape[1]
    current_cost, current_path = state
    current_city = current_path[-1]

    for next_city in range(cities_count):
        weight = graph[current_city][next_city]
        if next_city in current_path or weight == np.NINF:
            continue
        next_state = (current_cost+weight, current_path + [next_city])
        next_states.append(next_state)
    return next_states


def _get_next_states_final_step(graph, state):
    current_cost, current_path = state
    first_city = current_path[0]
    current_city = current_path[-1]

    cost = graph[current_city][first_city]

    if cost == np.NINF:
        return []

    new_state = (current_cost + cost, current_path + [first_city])
    return [new_state]
