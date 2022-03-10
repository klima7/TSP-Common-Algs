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

    for i in range(cities_count - 1):

        nearest_city = _find_city_nearest_to_path(graph, path)

        if not nearest_city:
            return

        path = _get_path_with_inserted_city(graph, path, nearest_city)

        if path is None:
            return

    cost = _get_path_cost(graph, path)
    return path, cost


def _find_city_nearest_to_path(graph, path):
    cities_count = graph.shape[0]
    not_visited_cities = [i for i in range(cities_count) if i not in path]
    cities_distances = []
    for target_city in not_visited_cities:
        minimum_distance = math.inf
        for start_city in path:
            distance = graph[start_city, target_city]
            if distance != -math.inf:
                minimum_distance = min(minimum_distance, distance)
        if minimum_distance != math.inf:
            city_distance = (minimum_distance, target_city)
            cities_distances.append(city_distance)
    if not cities_distances:
        return None
    nearest_city = sorted(cities_distances)[0][1]
    return nearest_city


def _get_path_with_inserted_city(graph, path, city):
    all_paths = [path[:pos] + [city] + path[pos:] for pos in range(1, len(path))]
    lengths_and_paths = [(_get_path_cost(graph, path), path) for path in all_paths]
    lengths_and_paths = [length_and_path for length_and_path in lengths_and_paths if length_and_path[0] != -math.inf]
    shortest_path = sorted(lengths_and_paths)[0][1] if lengths_and_paths else None
    return shortest_path


def _get_path_cost(graph, path):
    total_cost = 0
    for current_city, next_city in zip(path, path[1:]):
        cost = graph[current_city, next_city]
        total_cost += cost
    return total_cost
