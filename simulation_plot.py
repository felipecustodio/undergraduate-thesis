import matplotlib.pyplot as plt
import plotly.graph_objects as go
from config import agent_colors


def plot_population_line(dataframe):
    agent_options = [
        "Coward",
        "Impostor",
        "Altruist",
        "GreenBeardAltruist",
        "BeardlessGreenBeardAltruist",
    ]

    topology = dataframe["graph_config_topology"][0]

    fig = go.Figure()

    fig.update_layout(
        title=f"{topology} - Population size evolution",
        xaxis_title="Simulation step",
        yaxis_title="Population size",
        legend_title="Agents",
        # font=dict(
        #     family="Courier New, monospace",
        #     size=18,
        #     color="RebeccaPurple"
        # )
    )

    x_axis = list(range(max(dataframe["simulation step"]) + 1))
    for agent in agent_options:
        if not ((dataframe[f"agents_{agent}"] == 0).all()):
            average = dataframe.groupby(["simulation step"])[
                f"{agent} population size"
            ].mean()
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=average,
                    line=dict(color=agent_colors[agent]),
                    name=agent,
                )
            )

    fig.update_layout(legend=dict(y=0.5, font_size=16), hovermode="x")
    return fig


def plot_population_area(dataframe):
    agent_options = [
        "Coward",
        "Impostor",
        "Altruist",
        "GreenBeardAltruist",
        "BeardlessGreenBeardAltruist",
    ]

    topology = dataframe["graph_config_topology"][0]

    fig = go.Figure()

    fig.update_layout(
        title=f"{topology} - Population proportion evolution",
        xaxis_title="Simulation step",
        yaxis_title="Population %",
        legend_title="Agents",
        # font=dict(
        #     family="Courier New, monospace",
        #     size=18,
        #     color="RebeccaPurple"
        # )
    )

    x_axis = list(range(max(dataframe["simulation step"]) + 1))

    for agent in agent_options:
        if not ((dataframe[f"agents_{agent}"] == 0).all()):
            average = dataframe.groupby(["simulation step"])[
                f"{agent} population size"
            ].mean()
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=average,
                    # label=f"{agent} population",
                    name=agent,
                    mode="lines",
                    line=dict(color=agent_colors[agent], width=0.5),
                    stackgroup="one",
                    groupnorm="percent",  # sets the normalization for the sum of the stackgroup
                )
            )

    return fig


def plot_all_line(dataframe):
    agent_options = [
        "Coward",
        "Impostor",
        "Altruist",
        "GreenBeardAltruist",
        "BeardlessGreenBeardAltruist",
    ]

    fig = go.Figure()

    # plot averages
    x_axis = list(range(max(dataframe["simulation step"]) + 1))
    for agent in agent_options:
        if not ((dataframe[f"agents_{agent}"] == 0).all()):
            average = dataframe.groupby(["simulation step"])[
                f"{agent} population size"
            ].mean()
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=average,
                    name=f"{agent} - average",
                    line=dict(color=agent_colors[agent], width=4),
                )
            )

    for simulation in range(max(dataframe["simulation #"])):
        view = dataframe[dataframe["simulation #"] == simulation]
        for agent in agent_options:
            if not ((dataframe[f"agents_{agent}"] == 0).all()):
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=view[f"{agent} population size"],
                        name=f"{agent} - runs",
                        line=dict(color=agent_colors[agent], width=1, dash="dash"),
                    )
                )

    return fig


def plot_all_area(dataframe):
    agent_options = [
        "Coward",
        "Impostor",
        "Altruist",
        "GreenBeardAltruist",
        "BeardlessGreenBeardAltruist",
    ]

    fig = go.Figure()

    x_axis = list(range(max(dataframe["simulation step"]) + 1))

    for agent in agent_options:
        if not ((dataframe[f"agents_{agent}"] == 0).all()):
            average = dataframe.groupby(["simulation step"])[
                f"{agent} population size"
            ].mean()
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=average,
                    # label=f"{agent} population",
                    name=agent,
                    mode="lines",
                    line=dict(color=agent_colors[agent], width=0.5),
                    stackgroup="one",
                    groupnorm="percent",  # sets the normalization for the sum of the stackgroup
                )
            )

    for simulation in range(max(dataframe["simulation #"])):
        view = dataframe[dataframe["simulation #"] == simulation]
        for agent in agent_options:
            if not ((dataframe[f"agents_{agent}"] == 0).all()):
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=view[f"{agent} population size"],
                        name=f"{agent} - runs",
                        line=dict(color=agent_colors[agent], width=1, dash="dash"),
                    )
                )

    return fig


def plot_histogram(dataframe):
    pass


def plot_network_heatmap(G):
    pass
