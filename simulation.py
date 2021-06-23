import random

import numpy as np
from rich.console import Console

import agents
import network_traversal
from agents import *

console = Console()


def simulation(parameters, G):
    # reset graph parameters
    # and count node types
    for node in G.nodes():
        G.nodes[node]["visits"] = 0
        G.nodes[node]["agents"] = []

    # unwrap parameters
    # simulation parameters
    simulation_number = parameters["simulation_number"]
    steps = parameters["steps"]

    # agents parameters
    all_agents = parameters["agents"]
    population_size = parameters["population_size"]
    probability_survival = parameters["probability_survival"]

    group_size = parameters["group_size"]

    # calculate how many agents of each type
    # will exist in the population
    distribution = np.random.choice(
        a=list(all_agents.keys()),
        size=population_size,
        p=list(all_agents.values()),
        replace=True,
    )

    # reset class counters
    agents.Agent._total_alive_counter = 0
    agents.Agent._total_counter = 0
    for agent in all_agents:
        globals()[agent]._counter = 0
        globals()[agent]._alive_counter = 0

    # initialize population of Agents
    population = []
    for element in distribution:
        population.append(globals()[element](probability_survival=probability_survival))

    # calculate path from starting node
    traversal = network_traversal.traversal_options[parameters["traversal"]]
    max_traversal_steps = parameters["max_traversal_steps"]

    # try to find a blank node for 'housing',
    # else will use node '0'
    start_node = network_traversal.starting_random(G)
    path = traversal(G, start_node, max_traversal_steps, steps=0, visited=[], path=[])

    for agent in population:
        agent.node = start_node
        G.nodes[start_node]["agents"].append(agent)

    # store results for analysis
    rows = []

    # run simulation
    for day in range(steps):
        row = [simulation_number, day]
        for agent in all_agents:
            row.append(globals()[agent]._alive_counter)
            row.append(globals()[agent]._counter)
        rows.append(row)

        # check if entire population went extinct
        if len(population) <= 0:
            break

        # feeding cycle: agents will roam the network looking for food
        # shuffle population, agents will walk in this new order
        random.shuffle(population)

        for agent in population:
            for traversal_steps, node in enumerate(path):
                # increase visit counter for current node
                G.nodes[node]["visits"] += 1
                if traversal_steps <= max_traversal_steps:
                    # check if node has food and
                    # is not full (a node can fit 'group_size' agents)
                    if (
                        G.nodes[node]["food"]
                        and len(G.nodes[node]["agents"]) < group_size
                    ):
                        G.nodes[agent.node]["agents"].remove(agent)
                        agent.node = node
                        agent.fed = True
                        G.nodes[node]["agents"].append(agent)
                        break  # stop walking

        # agents have been placed around the network,
        # those that haven't found any food will die of starvation
        for agent in population:
            if not agent.fed:
                agent.die()
                population.remove(agent)
                continue

            # fetch agents in current node
            node = agent.node
            pair = G.nodes[node]["agents"]

            # if agent is alone, it will survive,
            # else they'll behave according to the type of node they're in.
            if len(pair) > 1:
                actor = pair[0]
                other = pair[1]

                node = G.nodes[node]

                if node["predator"]:
                    actor.behavior(other)

                    if not actor.alive:
                        population.remove(actor)

                    if not other.alive:
                        population.remove(other)

        # reproduction cycle
        # reproduction of the survivors
        children = []
        for agent in population:
            # double check just to be safe
            if agent.alive:
                sons = agent.reproduce()
                # kill and remove original agent
                # add children to population
                agent.die()
                population.remove(agent)
                children.extend(sons)
        population.extend(children)

        # teleport everyone to start node
        # reset agents
        for agent in population:
            agent.node = start_node
            agent.fed = False
            G.nodes[start_node]["agents"].append(agent)

        # reset nodes
        for node in G.nodes():
            if node is not start_node:
                G.nodes[node]["agents"] = []

    # simulation finished, return data
    return rows
