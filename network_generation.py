import networkx as nx
import numpy as np
from config import node_colors


def full_2d_grid(nodes, edges, k_nearest, p_new_edge):
    # if nodes > 20:
    #     print("[bold red]Too many nodes for this type of graph![/]")
    #     raise ValueError("too many nodes")

    G = nx.grid_2d_graph(nodes, nodes, periodic=False)
    # connect diagonally
    distances = dict(nx.all_pairs_shortest_path_length(G))
    for nodeA in list(G.nodes()):
        for nodeB in list(G.nodes()):
            if nodeA != nodeB:
                distance = distances[nodeA][nodeB]
                if distance == 2 and not G.has_edge(nodeA, nodeB):
                    G.add_edge(nodeA, nodeB)

    network = nx.Graph()

    tuple_to_node = {}
    for index, node in enumerate(list(G.nodes())):
        tuple_to_node[node] = index
        network.add_node(index)

    for edge in G.edges:
        nodeA = tuple_to_node[edge[0]]
        nodeB = tuple_to_node[edge[1]]
        network.add_edge(nodeA, nodeB)

    return network


def full_3d_grid(nodes, edges, k_nearest, p_new_edge):
    # if nodes > 20:
    #     print("[bold red]Too many nodes for this type of graph![/]")
    #     raise ValueError("too many nodes")

    # nodes x nodes
    G = nx.grid_graph(dim=(nodes, nodes, nodes), periodic=False)
    # connect diagonally
    distances = dict(nx.all_pairs_shortest_path_length(G))
    for nodeA in list(G.nodes()):
        for nodeB in list(G.nodes()):
            if nodeA != nodeB:
                distance = distances[nodeA][nodeB]
                if distance == 2 and not G.has_edge(nodeA, nodeB):
                    G.add_edge(nodeA, nodeB)

    network = nx.Graph()

    tuple_to_node = {}
    for index, node in enumerate(list(G.nodes())):
        tuple_to_node[node] = index
        network.add_node(index)

    for edge in G.edges:
        nodeA = tuple_to_node[edge[0]]
        nodeB = tuple_to_node[edge[1]]
        network.add_edge(nodeA, nodeB)

    return network


def complete_graph(nodes, edges, k_nearest, p_new_edge):
    return nx.complete_graph(int(nodes))


def path_graph(nodes, edges, k_nearest, p_new_edge):
    return nx.path_graph(int(nodes))


def random_lobster(nodes, edges, k_nearest, p_new_edge):
    return nx.random_lobster(nodes, p_new_edge, p_new_edge / edges)


def powerlaw_cluster(nodes, edges, k_nearest, p_new_edge):
    return nx.powerlaw_cluster_graph(nodes, edges, p_new_edge)


def barabasi_albert(nodes, edges, k_nearest, p_new_edge):
    return nx.barabasi_albert_graph(nodes, edges)


def erdos_renyi(nodes, edges, k_nearest, p_new_edge):
    return nx.erdos_renyi_graph(nodes, p_new_edge)


def newman_watts_strogatz(nodes, edges, k_nearest, p_new_edge):
    return nx.newman_watts_strogatz_graph(nodes, k_nearest, p_new_edge)


def random_internet(nodes, edges, k_nearest, p_new_edge):
    return nx.random_internet_as_graph(nodes)


def enrich_graph(G, graph_config):
    probability_tree = graph_config["probability_tree"]
    probability_predator = graph_config["probability_predator"]
    probability_blank = graph_config["probability_blank"]

    df = nx.to_pandas_edgelist(G)

    source_nodes = df["source"]
    target_nodes = df["target"]

    all_nodes = set(source_nodes.append(target_nodes))
    all_edges = [(row[1]["source"], row[1]["target"]) for row in df.iterrows()]

    # network will be the final, enriched structure
    network = nx.Graph()

    probabilities = [probability_tree, probability_predator, probability_blank]
    categories = np.random.choice(
        a=["tree", "predator", "blank"], size=len(all_nodes), p=probabilities
    )

    for index, node in enumerate(all_nodes):
        category = categories[index]

        food = False
        predator = False
        if category in ["tree", "predator"]:
            food = True
        if category == "predator":
            predator = True

        network.add_node(
            node,
            category=category,
            food=food,
            predator=predator,
            agents=[],
            visits=0,
            color=node_colors[category],
        )

    network.add_edges_from(all_edges)
    return network


generators = {
    # novel graphs
    "2D Full Grid": full_2d_grid,
    "3D Full Grid": full_3d_grid,
    "Complete Graph": complete_graph,
    "Path Graph": path_graph,
    # random networks
    "Random Lobster": random_lobster,
    "Barabasi-Albert": barabasi_albert,
    "Erdos-Renyi": erdos_renyi,
    "Newman-Watts-Strogatz": newman_watts_strogatz,
    # real world
    "Internet Autonomous System": random_internet,
}


def generate_graph(graph_config):
    nodes = graph_config["nodes"]
    edges = graph_config["edges"]
    k_nearest = graph_config["k_nearest"]
    p_new_edge = graph_config["p_new_edge"]

    topology = graph_config["topology"]
    generator_function = generators[topology]

    # enrich generated graph
    G = generator_function(nodes, edges, k_nearest, p_new_edge)
    network = enrich_graph(G, graph_config)
    return network
