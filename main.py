# -*- coding: utf-8 -*-

# my modules
# import agents
# import config
# import network_description
import network_generation

# import network_traversal
import network_plot
import simulation
import simulation_plot

import pandas as pd

from rich import print

# from rich.panel import Panel
# from rich.pretty import Pretty
from rich.console import Console
import rich.traceback

from tqdm import trange

rich.traceback.install(show_locals=False)
console = Console()


def run_simulation(parameters):
    print("[bold]Simulation")

    print(f"✸ Generating graph using {parameters['graph_config']['topology']}...")
    G = network_generation.generate_graph(parameters["graph_config"])

    # pretty = Pretty(parameters)
    # print(Panel.fit(pretty, title="Parameters"))

    results = []

    # print("[bold]Network descriptive statistics:")
    # print(network_statistics(G))
    # plot_network_characteristics(G)

    sims = parameters["sims"]

    for simulation_number in trange(sims, desc="✸ Running simulations"):
        parameters["simulation_number"] = simulation_number

        if not parameters["hub"]["use"]:
            current_results = simulation.simulation(parameters, G)
        else:
            current_results = simulation.simulation_hub(parameters, G)

        results.extend(current_results)

    # results = list of rows
    # insert results in a DataFrame
    print("✸ Building results DataFrame...")
    columns = ["simulation #", "simulation step"]
    for agent in parameters["agents"]:
        columns.append(f"{agent} population size")
        columns.append(f"{agent} lifetime size")
    df = pd.DataFrame(results, columns=columns)

    parameters_df = pd.json_normalize(parameters, sep="_")
    final = pd.concat([parameters_df, df], axis=1)
    final = final.fillna(method="ffill")
    final = final.drop(["simulation_number"], axis=1)
    final = final.sort_values(by=["simulation #", "simulation step"])

    print("✸ Building plots...")
    plots = {
        "population line": simulation_plot.plot_population_line(final),
        "population area": simulation_plot.plot_population_area(final),
        "network 2D": network_plot.plot_network_2D(G, parameters["graph_config"]),
        "network 3D": network_plot.plot_network_3D(G, parameters["graph_config"]),
    }

    return final, plots


"""
1 - Influência da topologia:
Cooperadores X Covardes: analisar o gráfico
temporal da fração dessas populações ao longo do tempo.
Figura do tempo a partir de 2: 45s no vídeo.
Simulamos nos modelos: (todas com o mesmo número de vértices e grau médio):
ER, BA, Small - world(p=0.01), Small - world(p=0.1), teremos uma curva para
cada um deles. Nesse caso, podemos colocar as curvas de cada modelo em um mesmo
plot, com a média e desvio padrão de umas 30 simulações.
"""

console.rule("[bold]Experimento 1")
print("Influência da topologia")

dfs = []

parameters = {
    "simulation_number": 0,
    "sims": 100,
    "steps": 50,
    "population_size": 1000,
    "group_size": 2,
    "traversal": "BFS",
    "max_traversal_steps": 50,
    "agents": {
        "Coward": 0.5,
        "Impostor": 0.0,
        "Altruist": 0.5,
        "GreenBeardAltruist": 0.0,
        "BeardlessGreenBeardAltruist": 0.0,
    },
    "probability_survival": 0.5,
    "graph_config": {
        "nodes": 500,
        "edges": 200,
        "k_nearest": 2,
        "p_new_edge": 0.1,
        "topology": "2D Full Grid",
        "probability_tree": 0.5,
        "probability_predator": 0.5,
        "probability_blank": 0.0,
    },
    "hub": {"use": False, "desc": True},
}

""" Control - 2D Grid """
print("Control - 2D grid")

parameters["graph_config"]["nodes"] = 20
df, plots = run_simulation(parameters)

dfs.append(df)

with open("results/1_2D.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/1_2D_population_line.png")
plots["population area"].write_image("plots/1_2D_population_area.png")
plots["network 2D"].write_image("plots/1_2D_network.png")
plots["network 3D"].write_image("plots/1_2D_network_3D.png")

"""ER"""
print("ER")

parameters["graph_config"]["nodes"] = 500
parameters["graph_config"]["topology"] = "Erdos-Renyi"
df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/1_ER.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/1_ER_population_line.png")
plots["population area"].write_image("plots/1_ER_population_area.png")
plots["network 2D"].write_image("plots/1_ER_network.png")
plots["network 3D"].write_image("plots/1_ER_network_3D.png")

"""BA"""
print("BA")

parameters["graph_config"]["topology"] = "Barabasi-Albert"
df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/1_BA.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/1_BA_population_line.png")
plots["population area"].write_image("plots/1_BA_population_area.png")
plots["network 2D"].write_image("plots/1_BA_network.png")
plots["network 3D"].write_image("plots/1_BA_network_3D.png")

"""Small-World $(p=0.01)$"""
print("Small-World (p=0.01)")

parameters["graph_config"]["topology"] = "Newman-Watts-Strogatz"
parameters["graph_config"]["p_new_edge"] = 0.01

df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/1_SW_001.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/1_SW_001_population_line.png")
plots["population area"].write_image("plots/1_SW_001_population_area.png")
plots["network 2D"].write_image("plots/1_SW_001_network.png")
plots["network 3D"].write_image("plots/1_SW_001_network_3D.png")

"""Small-World $(p = 0.1)$"""
print("Small-World (p = 0.1)")

parameters["graph_config"]["topology"] = "Newman-Watts-Strogatz"
parameters["graph_config"]["p_new_edge"] = 0.1

df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/1_SW_01.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/1_SW_01_population_line.png")
plots["population area"].write_image("plots/1_SW_01_population_area.png")
plots["network 2D"].write_image("plots/1_SW_01_network.png")
plots["network 3D"].write_image("plots/1_SW_01_network_3D.png")

all_results = pd.concat(dfs)
with open("results/1_all.csv", newline="", encoding="utf-8", mode="w") as fp:
    all_results.to_csv(fp, index=False)

"""
2 - Hubs altruístas:
a) Ordenar os vértices pelo grau e selecionar os mais conectados como sendo altruístas.
b) Ordenar os vértices e colocar os menos conectados como altruístas (inverso do caso interior).
c) Selecionar os altruístas de forma aleatória.
Obter as curvas de cooperadores X Covardes ao longo do tempo para cada um desses casos (apenas redes BA).
Plotar esses casos em um mesmo gráfico.
"""

# console.rule("[bold]Experimento 2")

"""
3 - influência da probabilidade de morte dos altruístas:
a) Repetir os casos 1 e 2 para probabilidades de morte 10%, 30% e 50%.
b) Rode para muitas simulações, talvez 10k passos e verifique a fração final de altruístas.
Construa um gráfico da fração final de altruístas (eixo y) versus a probabilidade de sobrevivência (eixo x).
Faça essa curva para as redes ER e BA.
"""

dfs = []

console.rule("[bold]Experimento 3")
print("Influência da probabilidade de morte dos altruístas")

parameters = {
    "simulation_number": 0,
    "sims": 100,
    "steps": 50,
    "population_size": 1000,
    "group_size": 2,
    "traversal": "BFS",
    "max_traversal_steps": 50,
    "agents": {
        "Coward": 0.5,
        "Impostor": 0.0,
        "Altruist": 0.5,
        "GreenBeardAltruist": 0.0,
        "BeardlessGreenBeardAltruist": 0.0,
    },
    "probability_survival": 0.5,
    "graph_config": {
        "nodes": 500,
        "edges": 200,
        "k_nearest": 2,
        "p_new_edge": 0.1,
        "topology": "2D Full Grid",
        "probability_tree": 0.5,
        "probability_predator": 0.5,
        "probability_blank": 0.0,
    },
    "hub": {"use": False, "desc": True},
}

""" p(survival) = 0.1 """
parameters["probability_survival"] = 0.1

""" Control - 2D Grid """
print("Control - 2D grid")

parameters["graph_config"]["nodes"] = 20
df, plots = run_simulation(parameters)

dfs.append(df)

with open("results/3_p01_2D.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/3_p01_2D_population_line.png")
plots["population area"].write_image("plots/3_p01_2D_population_area.png")
plots["network 2D"].write_image("plots/3_p01_2D_network.png")
plots["network 3D"].write_image("plots/3_p01_2D_network_3D.png")

"""ER"""
print("ER")

parameters["graph_config"]["nodes"] = 500
parameters["graph_config"]["topology"] = "Erdos-Renyi"
df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/3_p01_ER.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/3_p01_ER_population_line.png")
plots["population area"].write_image("plots/3_p01_ER_population_area.png")
plots["network 2D"].write_image("plots/3_p01_ER_network.png")
plots["network 3D"].write_image("plots/3_p01_ER_network_3D.png")

"""BA"""
print("BA")

parameters["graph_config"]["topology"] = "Barabasi-Albert"
df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/3_p01_BA.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/3_p01_BA_population_line.png")
plots["population area"].write_image("plots/3_p01_BA_population_area.png")
plots["network 2D"].write_image("plots/3_p01_BA_network.png")
plots["network 3D"].write_image("plots/3_p01_BA_network_3D.png")

"""Small-World $(p=0.01)$"""
print("Small-World (p=0.01)")

parameters["graph_config"]["topology"] = "Newman-Watts-Strogatz"
parameters["graph_config"]["p_new_edge"] = 0.01

df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/3_p01_SW_001.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/3_p01_SW_001_population_line.png")
plots["population area"].write_image("plots/3_p01_SW_001_population_area.png")
plots["network 2D"].write_image("plots/3_p01_SW_001_network.png")
plots["network 3D"].write_image("plots/3_p01_SW_001_network_3D.png")

"""Small-World $(p = 0.1)$"""
print("Small-World (p = 0.1)")

parameters["graph_config"]["topology"] = "Newman-Watts-Strogatz"
parameters["graph_config"]["p_new_edge"] = 0.1

df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/3_p01_SW_01.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/3_p01_SW_01_population_line.png")
plots["population area"].write_image("plots/3_p01_SW_01_population_area.png")
plots["network 2D"].write_image("plots/3_p01_SW_01_network.png")
plots["network 3D"].write_image("plots/3_p01_SW_01_network_3D.png")


""" p(survival) = 0.3 """
parameters["probability_survival"] = 0.3

""" Control - 2D Grid """
print("Control - 2D grid")

parameters["graph_config"]["nodes"] = 20
df, plots = run_simulation(parameters)

dfs.append(df)

with open("results/3_p03_2D.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/3_p03_2D_population_line.png")
plots["population area"].write_image("plots/3_p03_2D_population_area.png")
plots["network 2D"].write_image("plots/3_p03_2D_network.png")
plots["network 3D"].write_image("plots/3_p03_2D_network_3D.png")

"""ER"""
print("ER")

parameters["graph_config"]["nodes"] = 500
parameters["graph_config"]["topology"] = "Erdos-Renyi"
df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/3_p03_ER.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/3_p03_ER_population_line.png")
plots["population area"].write_image("plots/3_p03_ER_population_area.png")
plots["network 2D"].write_image("plots/3_p03_ER_network.png")
plots["network 3D"].write_image("plots/3_p03_ER_network_3D.png")

"""BA"""
print("BA")

parameters["graph_config"]["topology"] = "Barabasi-Albert"
df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/3_p03_BA.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/3_p03_BA_population_line.png")
plots["population area"].write_image("plots/3_p03_BA_population_area.png")
plots["network 2D"].write_image("plots/3_p03_BA_network.png")
plots["network 3D"].write_image("plots/3_p03_BA_network_3D.png")

"""Small-World $(p=0.01)$"""
print("Small-World (p=0.01)")

parameters["graph_config"]["topology"] = "Newman-Watts-Strogatz"
parameters["graph_config"]["p_new_edge"] = 0.01

df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/3_p03_SW_001.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/3_p03_SW_001_population_line.png")
plots["population area"].write_image("plots/3_p03_SW_001_population_area.png")
plots["network 2D"].write_image("plots/3_p03_SW_001_network.png")
plots["network 3D"].write_image("plots/3_p03_SW_001_network_3D.png")

"""Small-World $(p = 0.1)$"""
print("Small-World (p = 0.1)")

parameters["graph_config"]["topology"] = "Newman-Watts-Strogatz"
parameters["graph_config"]["p_new_edge"] = 0.1

df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/3_p03_SW_01.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/3_p03_SW_01_population_line.png")
plots["population area"].write_image("plots/3_p03_SW_01_population_area.png")
plots["network 2D"].write_image("plots/3_p03_SW_01_network.png")
plots["network 3D"].write_image("plots/3_p03_SW_01_network_3D.png")


""" p(survival) = 0.5 """
parameters["probability_survival"] = 0.5

""" Control - 2D Grid """
print("Control - 2D grid")

parameters["graph_config"]["nodes"] = 20
df, plots = run_simulation(parameters)

dfs.append(df)

with open("results/3_p05_2D.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/3_p05_2D_population_line.png")
plots["population area"].write_image("plots/3_p05_2D_population_area.png")
plots["network 2D"].write_image("plots/3_p05_2D_network.png")
plots["network 3D"].write_image("plots/3_p05_2D_network_3D.png")

"""ER"""
print("ER")

parameters["graph_config"]["nodes"] = 500
parameters["graph_config"]["topology"] = "Erdos-Renyi"
df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/3_p05_ER.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/3_p05_ER_population_line.png")
plots["population area"].write_image("plots/3_p05_ER_population_area.png")
plots["network 2D"].write_image("plots/3_p05_ER_network.png")
plots["network 3D"].write_image("plots/3_p05_ER_network_3D.png")

"""BA"""
print("BA")

parameters["graph_config"]["topology"] = "Barabasi-Albert"
df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/3_p05_BA.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/3_p05_BA_population_line.png")
plots["population area"].write_image("plots/3_p05_BA_population_area.png")
plots["network 2D"].write_image("plots/3_p05_BA_network.png")
plots["network 3D"].write_image("plots/3_p05_BA_network_3D.png")

"""Small-World $(p=0.01)$"""
print("Small-World (p=0.01)")

parameters["graph_config"]["topology"] = "Newman-Watts-Strogatz"
parameters["graph_config"]["p_new_edge"] = 0.01

df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/3_p05_SW_001.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/3_p05_SW_001_population_line.png")
plots["population area"].write_image("plots/3_p05_SW_001_population_area.png")
plots["network 2D"].write_image("plots/3_p05_SW_001_network.png")
plots["network 3D"].write_image("plots/3_p05_SW_001_network_3D.png")

"""Small-World $(p = 0.1)$"""
print("Small-World (p = 0.1)")

parameters["graph_config"]["topology"] = "Newman-Watts-Strogatz"
parameters["graph_config"]["p_new_edge"] = 0.1

df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/3_p05_SW_01.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/3_p05_SW_01_population_line.png")
plots["population area"].write_image("plots/3_p05_SW_01_population_area.png")
plots["network 2D"].write_image("plots/3_p05_SW_01_network.png")
plots["network 3D"].write_image("plots/3_p05_SW_01_network_3D.png")


""" p(survival) = 1.0 """
parameters["probability_survival"] = 1.0

""" Control - 2D Grid """
print("Control - 2D grid")

parameters["graph_config"]["nodes"] = 20
df, plots = run_simulation(parameters)

dfs.append(df)

with open("results/3_p10_2D.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/3_p10_2D_population_line.png")
plots["population area"].write_image("plots/3_p10_2D_population_area.png")
plots["network 2D"].write_image("plots/3_p10_2D_network.png")
plots["network 3D"].write_image("plots/3_p10_2D_network_3D.png")


"""ER"""
print("ER")

parameters["graph_config"]["nodes"] = 500
parameters["graph_config"]["topology"] = "Erdos-Renyi"
df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/3_p10_ER.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/3_p10_ER_population_line.png")
plots["population area"].write_image("plots/3_p10_ER_population_area.png")
plots["network 2D"].write_image("plots/3_p10_ER_network.png")
plots["network 3D"].write_image("plots/3_p10_ER_network_3D.png")


"""BA"""
print("BA")

parameters["graph_config"]["topology"] = "Barabasi-Albert"
df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/3_p10_BA.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/3_p10_BA_population_line.png")
plots["population area"].write_image("plots/3_p10_BA_population_area.png")
plots["network 2D"].write_image("plots/3_p10_BA_network.png")
plots["network 3D"].write_image("plots/3_p10_BA_network_3D.png")


"""Small-World $(p=0.01)$"""
print("Small-World (p=0.01)")

parameters["graph_config"]["topology"] = "Newman-Watts-Strogatz"
parameters["graph_config"]["p_new_edge"] = 0.01

df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/3_p10_SW_001.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/3_p10_SW_001_population_line.png")
plots["population area"].write_image("plots/3_p10_SW_001_population_area.png")
plots["network 2D"].write_image("plots/3_p10_SW_001_network.png")
plots["network 3D"].write_image("plots/3_p10_SW_001_network_3D.png")

########################################################

"""Small-World $(p = 0.1)$"""
print("Small-World (p = 0.1)")

parameters["graph_config"]["topology"] = "Newman-Watts-Strogatz"
parameters["graph_config"]["p_new_edge"] = 0.1

df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/3_p10_SW_01.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/3_p10_SW_01_population_line.png")
plots["population area"].write_image("plots/3_p10_SW_01_population_area.png")
plots["network 2D"].write_image("plots/3_p10_SW_01_network.png")
plots["network 3D"].write_image("plots/3_p10_SW_01_network_3D.png")

all_results = pd.concat(dfs)
with open("results/3_all.csv", newline="", encoding="utf-8", mode="w") as fp:
    all_results.to_csv(fp, index=False)


"""
4 - Simular o caso com green beard, repetindo as análises anteriores.
Você pode obter a curva para altruístas e green beard, como no minuto 11:30 do video no youtube.
"""

console.rule("[bold]Experimento 4")
print("Repetir experimentos 1 e 3 para Green Beards")

########################################################

print("Experimento 4 - Repetir experimento 1")
dfs = []

parameters = {
    "simulation_number": 0,
    "sims": 100,
    "steps": 50,
    "population_size": 1000,
    "group_size": 2,
    "traversal": "BFS",
    "max_traversal_steps": 50,
    "agents": {
        "Coward": 0.5,
        "Impostor": 0.0,
        "Altruist": 0.0,
        "GreenBeardAltruist": 0.5,
        "BeardlessGreenBeardAltruist": 0.0,
    },
    "probability_survival": 0.5,
    "graph_config": {
        "nodes": 500,
        "edges": 200,
        "k_nearest": 2,
        "p_new_edge": 0.1,
        "topology": "2D Full Grid",
        "probability_tree": 0.5,
        "probability_predator": 0.5,
        "probability_blank": 0.0,
    },
    "hub": {"use": False, "desc": True},
}

""" Control - 2D Grid """
print("Control - 2D grid")

parameters["graph_config"]["nodes"] = 20
df, plots = run_simulation(parameters)

dfs.append(df)

with open("results/4_1_2D.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_1_2D_population_line.png")
plots["population area"].write_image("plots/4_1_2D_population_area.png")
plots["network 2D"].write_image("plots/4_1_2D_network.png")
plots["network 3D"].write_image("plots/4_1_2D_network_3D.png")

"""ER"""
print("ER")

parameters["graph_config"]["nodes"] = 500
parameters["graph_config"]["topology"] = "Erdos-Renyi"
df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/4_1_ER.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_1_ER_population_line.png")
plots["population area"].write_image("plots/4_1_ER_population_area.png")
plots["network 2D"].write_image("plots/4_1_ER_network.png")
plots["network 3D"].write_image("plots/4_1_ER_network_3D.png")

"""BA"""
print("BA")

parameters["graph_config"]["topology"] = "Barabasi-Albert"
df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/4_1_BA.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_1_BA_population_line.png")
plots["population area"].write_image("plots/4_1_BA_population_area.png")
plots["network 2D"].write_image("plots/4_1_BA_network.png")
plots["network 3D"].write_image("plots/4_1_BA_network_3D.png")

"""Small-World $(p=0.01)$"""
print("Small-World (p=0.01)")

parameters["graph_config"]["topology"] = "Newman-Watts-Strogatz"
parameters["graph_config"]["p_new_edge"] = 0.01

df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/4_1_SW_001.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_1_SW_001_population_line.png")
plots["population area"].write_image("plots/4_1_SW_001_population_area.png")
plots["network 2D"].write_image("plots/4_1_SW_001_network.png")
plots["network 3D"].write_image("plots/4_1_SW_001_network_3D.png")

"""Small-World $(p = 0.1)$"""
print("Small-World (p = 0.1)")

parameters["graph_config"]["topology"] = "Newman-Watts-Strogatz"
parameters["graph_config"]["p_new_edge"] = 0.1

df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/4_1_SW_01.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_1_SW_01_population_line.png")
plots["population area"].write_image("plots/4_1_SW_01_population_area.png")
plots["network 2D"].write_image("plots/4_1_SW_01_network.png")
plots["network 3D"].write_image("plots/4_1_SW_01_network_3D.png")

all_results = pd.concat(dfs)
with open("results/4_1_all.csv", newline="", encoding="utf-8", mode="w") as fp:
    all_results.to_csv(fp, index=False)

########################################################

print("Experimento 4 - Repetir experimento 3")
dfs = []


""" p(survival) = 0.1 """
parameters["probability_survival"] = 0.1

""" Control - 2D Grid """
print("Control - 2D grid")

parameters["graph_config"]["nodes"] = 20
df, plots = run_simulation(parameters)

dfs.append(df)

with open("results/4_3_p01_2D.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_3_p01_2D_population_line.png")
plots["population area"].write_image("plots/4_3_p01_2D_population_area.png")
plots["network 2D"].write_image("plots/4_3_p01_2D_network.png")
plots["network 3D"].write_image("plots/4_3_p01_2D_network_3D.png")

"""ER"""
print("ER")

parameters["graph_config"]["nodes"] = 500
parameters["graph_config"]["topology"] = "Erdos-Renyi"
df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/4_3_p01_ER.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_3_p01_ER_population_line.png")
plots["population area"].write_image("plots/4_3_p01_ER_population_area.png")
plots["network 2D"].write_image("plots/4_3_p01_ER_network.png")
plots["network 3D"].write_image("plots/4_3_p01_ER_network_3D.png")

"""BA"""
print("BA")

parameters["graph_config"]["topology"] = "Barabasi-Albert"
df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/4_3_p01_BA.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_3_p01_BA_population_line.png")
plots["population area"].write_image("plots/4_3_p01_BA_population_area.png")
plots["network 2D"].write_image("plots/4_3_p01_BA_network.png")
plots["network 3D"].write_image("plots/4_3_p01_BA_network_3D.png")

"""Small-World $(p=0.01)$"""
print("Small-World (p=0.01)")

parameters["graph_config"]["topology"] = "Newman-Watts-Strogatz"
parameters["graph_config"]["p_new_edge"] = 0.01

df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/4_3_p01_SW_001.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_3_p01_SW_001_population_line.png")
plots["population area"].write_image("plots/4_3_p01_SW_001_population_area.png")
plots["network 2D"].write_image("plots/4_3_p01_SW_001_network.png")
plots["network 3D"].write_image("plots/4_3_p01_SW_001_network_3D.png")

"""Small-World $(p = 0.1)$"""
print("Small-World (p = 0.1)")

parameters["graph_config"]["topology"] = "Newman-Watts-Strogatz"
parameters["graph_config"]["p_new_edge"] = 0.1

df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/4_3_p01_SW_01.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_3_p01_SW_01_population_line.png")
plots["population area"].write_image("plots/4_3_p01_SW_01_population_area.png")
plots["network 2D"].write_image("plots/4_3_p01_SW_01_network.png")
plots["network 3D"].write_image("plots/4_3_p01_SW_01_network_3D.png")


""" p(survival) = 0.3 """
parameters["probability_survival"] = 0.3

""" Control - 2D Grid """
print("Control - 2D grid")

parameters["graph_config"]["nodes"] = 20
df, plots = run_simulation(parameters)

dfs.append(df)

with open("results/4_3_p03_2D.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_3_p03_2D_population_line.png")
plots["population area"].write_image("plots/4_3_p03_2D_population_area.png")
plots["network 2D"].write_image("plots/4_3_p03_2D_network.png")
plots["network 3D"].write_image("plots/4_3_p03_2D_network_3D.png")

"""ER"""
print("ER")

parameters["graph_config"]["nodes"] = 500
parameters["graph_config"]["topology"] = "Erdos-Renyi"
df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/4_3_p03_ER.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_3_p03_ER_population_line.png")
plots["population area"].write_image("plots/4_3_p03_ER_population_area.png")
plots["network 2D"].write_image("plots/4_3_p03_ER_network.png")
plots["network 3D"].write_image("plots/4_3_p03_ER_network_3D.png")

"""BA"""
print("BA")

parameters["graph_config"]["topology"] = "Barabasi-Albert"
df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/4_3_p03_BA.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_3_p03_BA_population_line.png")
plots["population area"].write_image("plots/4_3_p03_BA_population_area.png")
plots["network 2D"].write_image("plots/4_3_p03_BA_network.png")
plots["network 3D"].write_image("plots/4_3_p03_BA_network_3D.png")

"""Small-World $(p=0.01)$"""
print("Small-World (p=0.01)")

parameters["graph_config"]["topology"] = "Newman-Watts-Strogatz"
parameters["graph_config"]["p_new_edge"] = 0.01

df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/4_3_p03_SW_001.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_3_p03_SW_001_population_line.png")
plots["population area"].write_image("plots/4_3_p03_SW_001_population_area.png")
plots["network 2D"].write_image("plots/4_3_p03_SW_001_network.png")
plots["network 3D"].write_image("plots/4_3_p03_SW_001_network_3D.png")

"""Small-World $(p = 0.1)$"""
print("Small-World (p = 0.1)")

parameters["graph_config"]["topology"] = "Newman-Watts-Strogatz"
parameters["graph_config"]["p_new_edge"] = 0.1

df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/4_3_p03_SW_01.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_3_p03_SW_01_population_line.png")
plots["population area"].write_image("plots/4_3_p03_SW_01_population_area.png")
plots["network 2D"].write_image("plots/4_3_p03_SW_01_network.png")
plots["network 3D"].write_image("plots/4_3_p03_SW_01_network_3D.png")


""" p(survival) = 0.5 """
parameters["probability_survival"] = 0.5

""" Control - 2D Grid """
print("Control - 2D grid")

parameters["graph_config"]["nodes"] = 20
df, plots = run_simulation(parameters)

dfs.append(df)

with open("results/4_3_p05_2D.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_3_p05_2D_population_line.png")
plots["population area"].write_image("plots/4_3_p05_2D_population_area.png")
plots["network 2D"].write_image("plots/4_3_p05_2D_network.png")
plots["network 3D"].write_image("plots/4_3_p05_2D_network_3D.png")

"""ER"""
print("ER")

parameters["graph_config"]["nodes"] = 500
parameters["graph_config"]["topology"] = "Erdos-Renyi"
df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/4_3_p05_ER.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_3_p05_ER_population_line.png")
plots["population area"].write_image("plots/4_3_p05_ER_population_area.png")
plots["network 2D"].write_image("plots/4_3_p05_ER_network.png")
plots["network 3D"].write_image("plots/4_3_p05_ER_network_3D.png")

"""BA"""
print("BA")

parameters["graph_config"]["topology"] = "Barabasi-Albert"
df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/4_3_p05_BA.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_3_p05_BA_population_line.png")
plots["population area"].write_image("plots/4_3_p05_BA_population_area.png")
plots["network 2D"].write_image("plots/4_3_p05_BA_network.png")
plots["network 3D"].write_image("plots/4_3_p05_BA_network_3D.png")

"""Small-World $(p=0.01)$"""
print("Small-World (p=0.01)")

parameters["graph_config"]["topology"] = "Newman-Watts-Strogatz"
parameters["graph_config"]["p_new_edge"] = 0.01

df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/4_3_p05_SW_001.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_3_p05_SW_001_population_line.png")
plots["population area"].write_image("plots/4_3_p05_SW_001_population_area.png")
plots["network 2D"].write_image("plots/4_3_p05_SW_001_network.png")
plots["network 3D"].write_image("plots/4_3_p05_SW_001_network_3D.png")

"""Small-World $(p = 0.1)$"""
print("Small-World (p = 0.1)")

parameters["graph_config"]["topology"] = "Newman-Watts-Strogatz"
parameters["graph_config"]["p_new_edge"] = 0.1

df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/4_3_p05_SW_01.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_3_p05_SW_01_population_line.png")
plots["population area"].write_image("plots/4_3_p05_SW_01_population_area.png")
plots["network 2D"].write_image("plots/4_3_p05_SW_01_network.png")
plots["network 3D"].write_image("plots/4_3_p05_SW_01_network_3D.png")


""" p(survival) = 1.0 """
parameters["probability_survival"] = 1.0

""" Control - 2D Grid """
print("Control - 2D grid")

parameters["graph_config"]["nodes"] = 20
df, plots = run_simulation(parameters)

dfs.append(df)

with open("results/4_3_p10_2D.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_3_p10_2D_population_line.png")
plots["population area"].write_image("plots/4_3_p10_2D_population_area.png")
plots["network 2D"].write_image("plots/4_3_p10_2D_network.png")
plots["network 3D"].write_image("plots/4_3_p10_2D_network_3D.png")

"""ER"""
print("ER")

parameters["graph_config"]["nodes"] = 500
parameters["graph_config"]["topology"] = "Erdos-Renyi"
df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/4_3_p10_ER.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_3_p10_ER_population_line.png")
plots["population area"].write_image("plots/4_3_p10_ER_population_area.png")
plots["network 2D"].write_image("plots/4_3_p10_ER_network.png")
plots["network 3D"].write_image("plots/4_3_p10_ER_network_3D.png")

"""BA"""
print("BA")

parameters["graph_config"]["topology"] = "Barabasi-Albert"
df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/4_3_p10_BA.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_3_p10_BA_population_line.png")
plots["population area"].write_image("plots/4_3_p10_BA_population_area.png")
plots["network 2D"].write_image("plots/4_3_p10_BA_network.png")
plots["network 3D"].write_image("plots/4_3_p10_BA_network_3D.png")

"""Small-World $(p=0.01)$"""
print("Small-World (p=0.01)")

parameters["graph_config"]["topology"] = "Newman-Watts-Strogatz"
parameters["graph_config"]["p_new_edge"] = 0.01

df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/4_3_p10_SW_001.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_3_p10_SW_001_population_line.png")
plots["population area"].write_image("plots/4_3_p10_SW_001_population_area.png")
plots["network 2D"].write_image("plots/4_3_p10_SW_001_network.png")
plots["network 3D"].write_image("plots/4_3_p10_SW_001_network_3D.png")

"""Small-World $(p = 0.1)$"""
print("Small-World (p = 0.1)")

parameters["graph_config"]["topology"] = "Newman-Watts-Strogatz"
parameters["graph_config"]["p_new_edge"] = 0.1

df, plots = run_simulation(parameters)
dfs.append(df)

with open("results/4_3_p10_SW_01.csv", newline="", encoding="utf-8", mode="w") as fp:
    df.to_csv(fp, index=False)

plots["population line"].write_image("plots/4_3_p10_SW_01_population_line.png")
plots["population area"].write_image("plots/4_3_p10_SW_01_population_area.png")
plots["network 2D"].write_image("plots/4_3_p10_SW_01_network.png")
plots["network 3D"].write_image("plots/4_3_p10_SW_01_network_3D.png")

all_results = pd.concat(dfs)
with open("results/4_3_all.csv", newline="", encoding="utf-8", mode="w") as fp:
    all_results.to_csv(fp, index=False)
