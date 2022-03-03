import numpy as np


def dfs(graph, start):
    return _generic_search(graph, start, 0)


def bfs(graph, start):
    return _generic_search(graph, start, -1)


def _generic_search(graph, start, pop_position):
    states = [(0, [start])]
    acceptable_states = []
    while states:
        state = states.pop(pop_position)

        if _is_final_state(graph, state):
            acceptable_states.append(state)
        else:
            next_states = _get_next_states(graph, state)
            states = [*next_states, *states]

    acceptable_states.sort()
    optimal_state = acceptable_states[0] if acceptable_states else None

    if optimal_state is None:
        return None

    optimal_cost, optimal_path = optimal_state
    return optimal_path, optimal_cost


def _is_final_state(graph, state):
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

