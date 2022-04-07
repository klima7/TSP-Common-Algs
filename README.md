# TSP Common Algs
Comparison of common algorithms to solve Traveling Salesman Problem.

Considered algorithms:
- BFS (Breadth First Search)
- DFS (Depth First Search)
- Nearest Neighbor
- Nearest Insertion
- A star (with admissible heuristic)
- A star (with inadmissible heuristic)

Considered scenarios:
- Full graph / some connections dropped
- Symmetric / Assymetric version

Symmetric vs. asymmetric:
- Symmetric - cost of traveling between two cities is equal to euclidean distance.
- Asymmetric - like in symmetric versions, but 10% of cost is added or subtracted according to whether he travels uphill or downhill.

# Running
`python3 test.py`
