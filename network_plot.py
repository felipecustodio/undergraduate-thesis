import matplotlib.pyplot as plt
from grave import plot_network
import plotly.graph_objects as go
import igraph as ig
import networkx as nx


# def node_colorer(node_attributes):
#     color = node_attributes["color"]
#     shape = "o"  # random.choice(['s', 'o', '^', 'v', '8'])
#     return {"color": color, "size": 80, "shape": shape}


# def font_styler(attributes):
#     return {"font_size": 8, "font_weight": 0.5, "font_color": "k"}


# def tiny_font_styler(attributes):
#     return {"font_size": 4, "font_weight": 0.5, "font_color": attributes["color"]}


# def pathological_edge_style(edge_attrs):
#     return {"color": "black"}


# def plot_network_2D(graph, graph_config):
#     fig, ax = plt.subplots()
#     fig.set_dpi(100.0)
#     fig.set_size_inches(8, 6)

#     plot_network(
#         graph,
#         ax=ax,
#         layout="kamada_kawai",
#         node_style=node_colorer,
#         # edge_style=pathological_edge_style,
#         # node_label_style=tiny_font_styler,
#         # edge_label_style=tiny_font_styler
#     )

#     return fig


def plot_network_2D(G, graph_config):
    N = len(list(G.nodes()))

    # layout in 3D using igraph
    H = ig.Graph.from_networkx(G)
    layout = H.layout("lgl")

    # get coordinates
    # nodes
    Xn = [layout[k][0] for k in range(N)]  # x-coordinates
    Yn = [layout[k][1] for k in range(N)]  # y-coordinates
    # Zn = [layout[k][2] for k in range(N)]  # z-coordinates
    Xe = []
    Ye = []
    # Ze = []

    # edges
    for edge in G.edges():
        # this can be a tuple (1,2)
        # or a tuple of tuples ((1,2), (3,4))
        # we only want to iterate when it's a tuple of tuples
        if isinstance(edge[0], tuple):
            edge = list(edge)
            for e in edge:
                source = e[0]
                target = e[1]
                Xe += [layout[source][0], layout[target][0], None]
                Ye += [layout[source][1], layout[target][1], None]
                # Ze += [layout[source][2], layout[target][2], None]
        else:
            source = edge[0]
            target = edge[1]
            Xe += [layout[source][0], layout[target][0], None]
            Ye += [layout[source][1], layout[target][1], None]
            # Ze += [layout[source][2], layout[target][2], None]

    node_colors = nx.get_node_attributes(G, "color")
    node_categories = nx.get_node_attributes(G, "category")

    colors = [value for _, value in node_colors.items()]
    labels = [value for _, value in node_categories.items()]

    # add traces to figure
    # edges
    trace1 = go.Scatter(
        # trace1=px.scatter_3d(
        x=Xe,
        y=Ye,
        # z=Ze,
        mode="lines",
        line={"color": "rgb(125,125,125)", "width": 1},
        hoverinfo="none",
    )

    # nodes
    trace2 = go.Scatter(
        x=Xn,
        y=Yn,
        # z=Zn,
        mode="markers",
        marker={
            "symbol": "circle",
            "size": 6,
            "line": {"color": " rgb(50,50,50)", "width": 0.5},
            "color": colors,
        },
        text=labels,
        name="nodes",
    )

    # setup layout, text and modes
    # nodes = graph_config["nodes"]
    # edges = graph_config["edges"]
    # k_nearest = graph_config["k_nearest"]
    # p_new_edge = graph_config["p_new_edge"]
    topology = graph_config["topology"]

    axis = {
        "showbackground": True,
        "showline": True,
        "zeroline": True,
        "showgrid": True,
        "showticklabels": False,
    }

    layout = go.Layout(
        title=f"Topology: {topology}",
        font_color="black",
        #  width = 1000,
        #  height = 1000,
        showlegend=False,
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            # zaxis=dict(axis),
        ),
        margin=dict(t=100),
        hovermode="closest",
        # annotations=[
        #     {
        #         "showarrow": False,
        #         "text": f"Nodes: {nodes}, Edges: {edges}, Knn: {k_nearest}, P(new edge) = {p_new_edge}",
        #         "xref": "paper",
        #         "yref": "paper",
        #         "x": 0,
        #         "y": 0.1,
        #         "xanchor": "left",
        #         "yanchor": "bottom",
        #         "font": {
        #             "size": 14,
        #         },
        #     }
        # ],
    )

    data = [trace1, trace2]

    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(legend=dict(y=0.5, font_size=16))
    # fig.show()
    return fig


def plot_network_3D(G, graph_config):
    N = len(list(G.nodes()))

    # layout in 3D using igraph
    H = ig.Graph.from_networkx(G)
    layout = H.layout("kk3d")

    # get coordinates
    # nodes
    Xn = [layout[k][0] for k in range(N)]  # x-coordinates
    Yn = [layout[k][1] for k in range(N)]  # y-coordinates
    Zn = [layout[k][2] for k in range(N)]  # z-coordinates
    Xe = []
    Ye = []
    Ze = []

    # edges
    for edge in G.edges():
        # this can be a tuple (1,2)
        # or a tuple of tuples ((1,2), (3,4))
        # we only want to iterate when it's a tuple of tuples
        if isinstance(edge[0], tuple):
            edge = list(edge)
            for e in edge:
                source = e[0]
                target = e[1]
                Xe += [layout[source][0], layout[target][0], None]
                Ye += [layout[source][1], layout[target][1], None]
                Ze += [layout[source][2], layout[target][2], None]
        else:
            source = edge[0]
            target = edge[1]
            Xe += [layout[source][0], layout[target][0], None]
            Ye += [layout[source][1], layout[target][1], None]
            Ze += [layout[source][2], layout[target][2], None]

    node_colors = nx.get_node_attributes(G, "color")
    node_categories = nx.get_node_attributes(G, "category")

    colors = [value for _, value in node_colors.items()]
    labels = [value for _, value in node_categories.items()]

    # add traces to figure
    # edges
    trace1 = go.Scatter3d(
        # trace1=px.scatter_3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode="lines",
        line={"color": "rgb(125,125,125)", "width": 1},
        hoverinfo="none",
    )

    # nodes
    trace2 = go.Scatter3d(
        x=Xn,
        y=Yn,
        z=Zn,
        mode="markers",
        marker={
            "symbol": "circle",
            "size": 6,
            "line": {"color": " rgb(50,50,50)", "width": 0.5},
            "color": colors,
        },
        text=labels,
        name="nodes",
    )

    # setup layout, text and modes
    # nodes = graph_config["nodes"]
    # edges = graph_config["edges"]
    # k_nearest = graph_config["k_nearest"]
    # p_new_edge = graph_config["p_new_edge"]
    topology = graph_config["topology"]

    axis = {
        "showbackground": True,
        "showline": True,
        "zeroline": True,
        "showgrid": True,
        "showticklabels": False,
    }

    layout = go.Layout(
        title=f"Topology: {topology}",
        font_color="black",
        #  width = 1000,
        #  height = 1000,
        showlegend=False,
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
        ),
        margin=dict(t=100),
        hovermode="closest",
        # annotations=[
        #     {
        #         "showarrow": False,
        #         "text": f"Nodes: {nodes}, Edges: {edges}, Knn: {k_nearest}, P(new edge) = {p_new_edge}",
        #         "xref": "paper",
        #         "yref": "paper",
        #         "x": 0,
        #         "y": 0.1,
        #         "xanchor": "left",
        #         "yanchor": "bottom",
        #         "font": {
        #             "size": 14,
        #         },
        #     }
        # ],
    )

    data = [trace1, trace2]

    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(legend=dict(y=0.5, font_size=16))
    # fig.show()
    return fig
