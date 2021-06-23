import networkx as nx
import igraph as ig
import numpy as np
import powerlaw
from matplotlib import pyplot as plt

"""Auxiliary functions"""


def ig_to_nx(graph):
    A = graph.get_edgelist()
    g = nx.Graph(A)
    return g


def nx_to_ig(graph):
    g = ig.Graph.TupleList(graph.edges(), directed=False)
    return g


def pearson(x, y):
    n = len(x)
    sum_x = float(sum(x))
    sum_y = float(sum(y))
    sum_x_sq = sum(map(lambda x: pow(x, 2), x))
    sum_y_sq = sum(map(lambda x: pow(x, 2), y))
    psum = sum(map(lambda x, y: x * y, x, y))
    num = psum - (sum_x * sum_y / n)
    den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)

    if den == 0:
        return 0
    return num / den


"""Network description functions"""


def stat_moment(graph, moment):
    measure = 0
    for node in graph.nodes():
        measure += graph.degree(node) ** moment
    return measure / graph.number_of_nodes()


def giant_component(graph):
    components = [graph.subgraph(c) for c in nx.connected_components(graph)]
    return nx.Graph(max((components), key=len))


def degree_distribution(graph):
    degrees = {}
    for node in graph.nodes():
        degree = graph.degree(node)
        if degree not in degrees:
            degrees[degree] = 0
        degrees[degree] += 1
    distribution = sorted(degrees.values())
    return distribution


def centrality_distribution(centrality):
    dists = {}
    for value in centrality.values():
        if value not in dists:
            dists[value] = 0
        dists[value] += 1
    return list(dists.values())


def clustering_distribution(graph):
    coefficients = list((nx.clustering(graph)).values())
    dist = {}
    for value in coefficients:
        if value not in dist:
            dist[value] = 0
        dist[value] += 1
    return list(dist.values())


def degree_distribution_coefficient(graph):
    distribution = degree_distribution(graph)
    distribution = [x for x in distribution if x != 0]

    np.seterr(divide="ignore", invalid="ignore")
    fit = powerlaw.Fit(distribution)
    coefficient = fit.power_law.alpha

    return coefficient


def entropy(graph):
    entropy = 0
    distribution = degree_distribution(graph)

    for value in distribution:
        if value > 0:
            val = value / graph.number_of_nodes()
            entropy -= (val) * math.log2(val)
    return entropy


def average_degree(graph):
    degrees = graph.degree().values()
    average = sum(degrees) / len(degrees)
    return average


def shortest_paths_distribution(graph):
    lengths = dict(nx.shortest_path_length(graph))
    frequencies = {}
    for source, targets in lengths.items():
        for target in targets:
            length = lengths[source][target]
            if length not in frequencies:
                frequencies[length] = 0
            frequencies[length] += 1

    return list(frequencies.values())


def assortativity(graph):
    assortativity = nx.degree_assortativity_coefficient(graph)
    return assortativity


def k_x_knn_correlation(graph):
    degrees = list(graph.degree().values())
    knn_degrees = list((nx.average_degree_connectivity(graph)).values())
    knn_vertex = list((nx.average_neighbor_degree(graph)).values())
    k = range(1, len(knn_degrees) + 1)

    # correlation k(x) x knn(x)
    correlation = pearson(graph.degree().values(), knn_vertex)
    return correlation


def modularities(graph):
    # convert to igraph
    g = nx_to_ig(graph)
    # edge betweenness centrality
    community_edge = (
        g.community_edge_betweenness(directed=False).as_clustering().modularity
    )
    # fast-greedy
    community_fg = g.community_fastgreedy().as_clustering().modularity
    # eigenvectors of matrices
    community_eigen = g.community_leading_eigenvector().modularity
    # walktrap
    community_walktrap = g.community_walktrap().as_clustering().modularity
    return community_edge, community_fg, community_eigen, community_walktrap


"""Plots"""


def centralities_histogram(graph):
    # centrality measures
    betweenness_centrality = centrality_distribution(nx.betweenness_centrality(graph))
    closeness_centrality = centrality_distribution(nx.closeness_centrality(graph))
    eigenvector_centrality = centrality_distribution(
        nx.eigenvector_centrality(graph, max_iter=1000)
    )
    pagerank = centrality_distribution(nx.pagerank(graph))

    # normalize
    betweenness_centrality = [
        x / max(betweenness_centrality) for x in betweenness_centrality
    ]
    closeness_centrality = [x / max(closeness_centrality) for x in closeness_centrality]
    eigenvector_centrality = [
        x / max(eigenvector_centrality) for x in eigenvector_centrality
    ]
    pagerank = [x / max(pagerank) for x in pagerank]

    # plot distributions
    plt.title("Medidas de centralidade")
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    ax0.plot(betweenness_centrality, marker="None")
    ax0.set_xlabel("Betweenness Centrality")
    ax1.plot(closeness_centrality, marker="None")
    ax1.set_xlabel("Closeness Centrality")
    ax2.plot(eigenvector_centrality, marker="None")
    ax2.set_xlabel("Eigenvector Centrality")
    ax3.plot(pagerank, marker="None")
    ax3.set_xlabel("PageRank")

    plt.legend(loc="upper right")
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    return fig


def shortest_paths_histogram(graph):
    fig = plt.figure()
    plt.title("Histograma de menores caminhos")
    plot = plt.subplot()
    distribution = shortest_paths_distribution(graph)
    # normalize
    distribution = [x / sum(distribution) for x in distribution]

    # plot
    x = np.linspace(0, nx.diameter(graph) + 1, len(distribution))
    plot.plot(x, distribution, marker="None")

    # style
    plot.spines["right"].set_visible(False)
    plot.spines["top"].set_visible(False)
    plot.yaxis.set_ticks_position("left")
    plot.xaxis.set_ticks_position("bottom")

    plt.xlabel("distance")
    plt.ylabel("probability")

    plt.legend(loc="upper right")
    plt.subplots_adjust(hspace=0.5)

    return fig


def clustering_histogram(graph):
    fig = plt.figure()
    plot = plt.subplot()
    plt.title("Distribuição acumulada do coeficiente de aglomeração local")

    distribution = clustering_distribution(graph)

    # normalize
    distribution = [x / sum(distribution) for x in distribution]

    # accumulated distribution
    distribution = np.cumsum(distribution)

    x = np.linspace(0, 1, len(distribution))
    plot.plot(x, distribution, color="#FF7676")

    # style
    plot.spines["right"].set_visible(False)
    plot.spines["top"].set_visible(False)
    plot.yaxis.set_ticks_position("left")
    plot.xaxis.set_ticks_position("bottom")

    plt.xlabel("coeficiente de aglomeração local")
    plt.ylabel("probabilidade")

    plt.legend(loc="lower right")
    plt.subplots_adjust(hspace=0.5)

    return fig


def plot_modularity_evolution(graph):
    # convert
    g = nx_to_ig(graph)

    # fast-greedy
    evolution = g.community_fastgreedy()
    count = evolution.optimal_count

    # aux
    count = count - 1
    tam = len(g.vs)

    # axis
    value_x = range(tam, count, -1)
    value_y = np.zeros(len(value_x))

    list_values_y = range(len(value_y))
    for i in list_values_y:
        value_y[i] = evolution.as_clustering(n=value_x[i]).modularity

    # reverse
    value_x = value_x[::-1]

    # plot
    fig = plt.figure()
    plt.plot(value_x, value_y, marker="o")
    plt.title("Modularity Evolution")
    plt.grid(False)
    plt.ylabel("Modularity")
    plt.xlabel("Step")

    return fig


def plot_k_knn(graph):
    degrees = list(dict(graph.degree()).values())
    knn_degrees = list((nx.average_degree_connectivity(graph)).values())
    knn_vertex = list((nx.average_neighbor_degree(graph)).values())
    k = range(1, len(knn_degrees) + 1)

    fig = plt.figure()
    plt.title("k(x) x knn(x)")
    ax1 = fig.add_subplot(111, label="k(x) x knn(x)")
    ax2 = fig.add_subplot(111, label="knn(k)", frame_on=False)

    # knn(x) - scatter
    plot1 = ax1.scatter(degrees, knn_vertex, s=10, marker="o", label="k(x) x knn(x)")
    ax1.set_xlabel("k(x)")
    ax1.set_ylabel("knn(x)")
    ax1.tick_params(axis="x")
    ax1.tick_params(axis="y")

    # knn(k) - line
    plot2 = ax2.plot(k, knn_degrees, label="knn(k)")
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.set_xlabel("k")
    ax2.set_ylabel("knn(k)")
    ax2.xaxis.set_label_position("top")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(axis="x")
    ax2.tick_params(axis="y")

    # plot
    fig.tight_layout()
    plt.figlegend((plot1, plot2), ("k(x) x knn(x)", "knn(k)"), loc="lower right")

    fig.subplots_adjust(top=0.85, bottom=0.15)
    plt.suptitle("k x knn - %s")
    plt.grid(False)

    return fig


"""Describe network"""


def network_statistics(graph):
    degrees = list(graph.degree())
    num_nodes = len(list(graph.nodes()))
    num_edges = len(list(graph.edges()))
    avg_degree = np.mean(np.array(degrees)[:, 1])
    med_degree = np.median(np.array(degrees)[:, 1])
    max_degree = max(np.array(degrees)[:, 1])
    min_degree = np.min(np.array(degrees)[:, 1])
    alpha = (
        powerlaw.Fit(centrality_distribution(nx.betweenness_centrality(graph)))
    ).alpha
    degree_coefficient = degree_distribution_coefficient(graph)
    community_edge, community_fg, community_eigen, community_walktrap = modularities(
        graph
    )

    scale_free = False
    if degree_coefficient >= 2 and degree_coefficient <= 3:
        scale_free = True

    powerlaw_betweenness = False
    if alpha >= 2 and alpha <= 3:
        powerlaw_betweenness = True

    return {
        "Number of nodes": num_nodes,
        "Number of edges": num_edges,
        "Average degree": avg_degree,
        "Median degree": med_degree,
        "Maximum degree": max_degree,
        "Minimum degree": min_degree,
        "Second statistical moment": stat_moment(graph, 2),
        "Average local clustering coefficient": nx.average_clustering(graph),
        "Transitivity": nx.transitivity(graph),
        "Average shortest path length": nx.average_shortest_path_length(graph),
        "Diameter": nx.diameter(graph),
        "Alpha (powerlaw fit)": alpha,
        "Betweenness Centrality obeys powerlaw": powerlaw_betweenness,
        "Shannon entropy": entropy(giant_component(graph)),
        "Scale free": scale_free,
        "Modularity - edge betweenness centrality": community_edge,
        "Modularity - fast greedy": community_fg,
        "Modularity - eigenvector centrality": community_eigen,
        "Modularity - walktrap": community_walktrap,
    }


def plot_network_characteristics(graph):
    figs = []
    figs.append(centralities_histogram(graph))
    figs.append(shortest_paths_histogram(graph))
    figs.append(clustering_histogram(graph))
    figs.append(plot_modularity_evolution(graph))
    figs.append(plot_k_knn(graph))
    return figs
