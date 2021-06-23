import numpy as np


def starting_random(G):
    blank_nodes = [
        node
        for node, attributes in G.nodes(data=True)
        if attributes["category"] == "blank"
    ]
    # if there are no blank nodes,
    # start at node 0
    if not blank_nodes:
        return 0

    return np.random.choice(a=blank_nodes, size=1)[0]


def traversal_random(G, node, max_steps, steps=0, visited=[], path=[]):
    while steps <= max_steps:
        path.append(node)
        neighbors = list(G.neighbors(node))
        node = np.random.choice(a=neighbors, size=1)[0]
        steps += 1

    return path


def traversal_bfs(G, node, max_steps, steps=0, visited=[], path=[]):
    queue = []
    visited.append(node)
    path.append(node)

    for neighbor in G.neighbors(node):
        if neighbor not in visited:
            # visit
            next_node = neighbor
            path.append(next_node)
            # go back
            path.append(node)
            # enqueue, but don't count as visited yet
            queue.append(neighbor)

    for enqueued_node in queue:
        traversal_bfs(G, enqueued_node, visited, path)
        queue.remove(enqueued_node)

    return path


def traversal_dfs(G, node, max_steps, steps=0, visited=[], path=[]):
    visited.append(node)
    path.append(node)

    for neighbor in G.neighbors(node):
        if neighbor not in visited:
            # visit
            next_node = neighbor
            traversal_dfs(G, next_node, visited, path)
            # go back
            path.append(node)

    return path


"""Dictionary of traversal algorithms and their display names"""

traversal_options = {
    "BFS": traversal_bfs,
    "DFS": traversal_dfs,
    "Random": traversal_random,
}
