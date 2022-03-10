from models import City
from graphs import generate_graph, check_connected
from search import dfs, bfs, greedy_search
import time


def display_cities(cities):
    print('* CITIES')
    for nr, city in enumerate(cities):
        print(f'{nr}: {city}')
    print()


def display_test_header(connections_drop, symmetric, search_method):
    method_name = search_method.__name__.upper()
    print(f'* TEST connections_drop: {connections_drop}; symmetric: {symmetric}; method: {method_name}')


def display_test_result(connected, result=None, duration=None):
    print(f'- Connected: {connected}')
    print(f'- Solved: {result is not None}')
    if result:
        path, cost = result
        print(f'- Path: {path}')
        print(f'- Cost: {round(cost, 5)}')
    if duration is not None:
        print(f'- Time: {round(duration, 5)}s')
    print()


def display_not_connected():
    print('- Graph is not connected\n')


def test(cities, start_city, connections_drop, symmetric, search_method, seed=None):
    display_test_header(connections_drop, symmetric, search_method)
    start_time = time.time()

    # 2. Represent the created map as a weighted, directed graph
    graph = generate_graph(cities, connections_drop=connections_drop, symmetric=symmetric, seed=seed)

    # Make sure whether generated graph is connected
    # This is necessary condition, but not sufficient
    # It may be still impossible to visit every city only once
    connected = check_connected(graph)
    if not connected:
        display_test_result(connected=False)
        return

    # 3. Search graph
    result = search_method(graph, start_city)

    end_time = time.time()
    display_test_result(connected=True, result=result, duration=end_time-start_time)


if __name__ == '__main__':
    seed = 222467
    cities_count = 8
    start_city = 0

    # 1. Create a set of cities
    cities = City.generate(count=cities_count, x_range=(-100, 100), y_range=(-100, 100), z_range=(0, 50), seed=seed)
    display_cities(cities)

    for connections_drop in [0.0, 0.2]:
        for symmetric in [True, False]:
            for method in [dfs, bfs, greedy_search]:
                test(cities, start_city, connections_drop, symmetric, method, seed=seed)