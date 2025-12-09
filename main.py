"""
Luis Palma: 58842
Rodrigo Lopes: 66
Diogo :

Create virtual enviorment:  py -3.10 -m venv venv
Activate VENV  .\venv\\Scripts\\Activate
Install requirements: pip install -r .\requirementsWin.txt

fonte: https://github.com/benedekrozemberczki/MUSAE

run: python main.py


Twitch Social Networks


These datasets used for node classification and transfer learning are Twitch user-user networks of gamers who stream in a certain language.
Nodes are the users themselves and the links are mutual friendships between them. Vertex features are extracted based on the games played and liked,
location and streaming habits. Datasets share the same set of node features, this makes transfer learning across networks possible.
These social networks were collected in May 2018.
The supervised task related to these networks is binary node classification - one has to predict whether a streamer uses explicit language.

"""

import sys
import numpy as np
import polars as pl
import seaborn as sns
import orjson
import networkx as nx
from collections import Counter
import plotly.express as px
import math
from matplotlib import pyplot as plt
import os
import random


"""
Exemplo de PTBR
features_file: data/PTBR/musae_PTBR_features.json
target_file:data/PTBR/musae_PTBR_target.csv
eadges file: data/PTBR/musae_PTBR_edges.csv
"""

""" Function for creating graph using the sample language like PTBR, DE, FR , RU, ENGB, ES 
requires: Three files 
    features_file: feature file in json format 
    target_files: target file in csv format 
    eadges_file: degree in csv format 
"""

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data(features_file, target_file, eadges_file):

    print("\n Loading features\n ")
    with open(features_file, "rb") as f:
        features_dict = orjson.loads(f.read())

    print("\n Loading edges and targets\n ")

    eadges = pl.read_csv(eadges_file)
    targets = pl.read_csv(target_file)

    print(" \n Creating graph \n ")

    eadges_tuples = list(zip(eadges["from"].to_list(), eadges["to"].to_list()))
    G = nx.Graph()
    G.add_edges_from(eadges_tuples)

    """
    Mapping "new_id" -> "mature" from target CSV
    """
    mature_map = dict(zip(targets["new_id"].to_list(), targets["mature"].to_list()))
    nx.set_node_attributes(G, mature_map, "mature")
    nx.set_node_attributes(G, features_dict, "features")

    print("\n Done !!!!\n")

    return G


def centrality_community(g):
    # --- 1. Centrality Features ---
    print("\n--- Centrality Features (Top 5) ---")

    # Calculate all centralities
    # Using max_iter=1000 for Eigenvector to ensure convergence
    try:
        eigenvector_cent = nx.eigenvector_centrality(g, max_iter=1000)
    except:
        # Fallback if it fails to converge (rare)
        eigenvector_cent = {n: 0.0 for n in g.nodes()}

    degree_cent = nx.degree_centrality(g)
    betweenness_cent = nx.betweenness_centrality(g)
    closeness_cent = nx.closeness_centrality(g)
    clustering_coeffs = nx.clustering(g)

    # Helper to get top 5
    def get_top_5(cent_dict):
        return sorted(cent_dict.items(), key=lambda x: x[1], reverse=True)[:5]

    top5_degree = get_top_5(degree_cent)
    top5_betweenness = get_top_5(betweenness_cent)
    top5_closeness = get_top_5(closeness_cent)
    top5_eigenvector = get_top_5(eigenvector_cent)
    top5_clustering = get_top_5(clustering_coeffs)

    # Create a nice DataFrame for display
    # We add the new columns here
    df_top5 = pl.DataFrame(
        {
            "Rank": [1, 2, 3, 4, 5],
            "Degree": [f"{n}: {s:.4f}" for n, s in top5_degree],
            "Betweenness": [f"{n}: {s:.4f}" for n, s in top5_betweenness],
            "Closeness": [f"{n}: {s:.4f}" for n, s in top5_closeness],
            "Eigenvector": [f"{n}: {s:.4f}" for n, s in top5_eigenvector],
            "Local Clustering": [f"{n}: {s:.4f}" for n, s in top5_clustering],
        }
    )

    print(df_top5)

    # --- 2. Community Detection ---
    print("\n--- Community Detection (Top 5 by Size) ---")

    try:
        communities_gmc = list(nx.community.greedy_modularity_communities(g))
    except AttributeError:
        communities_gmc = []

    try:
        if hasattr(nx.community, "louvain_communities"):
            communities_louvain = list(nx.community.louvain_communities(g))
        else:
            communities_louvain = []
    except AttributeError:
        communities_louvain = []

    # Lambda: Sorts communities by SIZE (len), returns (Index, Size)
    get_top_5_comms = lambda comms: sorted(
        [(i, len(c)) for i, c in enumerate(comms)], key=lambda x: x[1], reverse=True
    )[:5]

    top5_gmc = get_top_5_comms(communities_gmc)
    top5_louvain = get_top_5_comms(communities_louvain)

    # Pad with placeholders if < 5 found (Crucial for Polars DataFrame creation)
    while len(top5_gmc) < 5:
        top5_gmc.append(("-", 0))
    while len(top5_louvain) < 5:
        top5_louvain.append(("-", 0))

    df_comm = pl.DataFrame(
        {
            "Rank": range(1, 6),
            "Greedy Mod (ID: Size)": [f"{i}: {s}" for i, s in top5_gmc],
            "Louvain (ID: Size)": [f"{i}: {s}" for i, s in top5_louvain],
        }
    )

    print(df_comm)
    print(
        f"\nTotal Communities found -> GMC: {len(communities_gmc)}, Louvain: {len(communities_louvain)}"
    )

    # --- 3. Small World Properties ---
    print("\n--- Small World Properties ---")

    if nx.is_connected(g):
        G_lcc = g
    else:
        # Use Largest Connected Component
        G_lcc = g.subgraph(max(nx.connected_components(g), key=len))

    avg_clustering = nx.average_clustering(g)

    # Calculate Path Length only on LCC and if valid
    if len(G_lcc) > 1:
        avg_path_length = nx.average_shortest_path_length(G_lcc)
    else:
        avg_path_length = float("nan")

    # Sigma calculation
    if not math.isnan(avg_path_length) and avg_path_length != 0:
        sigma_simple = avg_clustering / avg_path_length
    else:
        sigma_simple = float("nan")

    df_sw = pl.DataFrame(
        {
            "Property": [
                "Avg Clustering Coefficient",
                "Avg Path Length (LCC)",
                "Simple Ratio (C/L)",
            ],
            "Value": [
                f"{avg_clustering:.4f}",
                f"{avg_path_length:.4f}" if not math.isnan(avg_path_length) else "N/A",
                f"{sigma_simple:.4f}" if not math.isnan(sigma_simple) else "N/A",
            ],
        }
    )

    print(df_sw)
    return


def ploting_distribuitions(G):

    def plot_degree_distribution(
        df, x_scale="linear", y_scale="linear", title="Degree Distribution"
    ):

        fig = px.scatter(
            df,
            x="Degree (k)",
            y="Count (P(k))",
            log_x=(x_scale == "log"),
            log_y=(y_scale == "log"),
            title=title,
            template="plotly_white",
        )
        fig.show()

    """ eadges count """
    degrees = [d for _n, d in G.degree()]
    degree_counts = Counter(degrees)

    # Prepare DataFrame for Plotly
    df_dist = pl.DataFrame(
        {
            "Degree (k)": list(degree_counts.keys()),
            "Count (P(k))": list(degree_counts.values()),
        }
    )
    """ plot 1: distribuição normal (Linear/Linear)"""

    plot_degree_distribution(
        df_dist,
        x_scale="linear",
        y_scale="linear",
        title="Degree Distribution (Normal)",
    )

    """ plot 2: distribuição logarítmica (Log/Log)

    Caso se verifique semelhante a uma linha reta, indica uma tendencia para uma power law  

    """

    plot_degree_distribution(df_dist, "log", "log", "Degree Distribution (Log-Log)")

    """ plot 3: distribuição semi-log (Linear/Log) 

    Se  o grafico log log é uma curva, no entanto este é uma linha. Pode se verificar que é exponencial 

    """

    plot_degree_distribution(
        df_dist, "linear", "log", "Degree Distribution (Semi-Log Y)"
    )


""" feature based  classification  """

"""
this function Analysis of features
Shows how the Homophily Analysis
shows the top 5 games for each type streamer "mature" and "Non-Mature"

"""


def deep_analise(G, features_file):
    def get_top_features(node_list, features_dict, top_n=5):
        games = []
        for n in node_list:
            games.extend(features_dict.get(str(n), []))
        counter = Counter(games)
        return counter.most_common(top_n)

    # Assortativity
    try:
        r = nx.attribute_assortativity_coefficient(G, "mature")
        print(f"\n--- Homophily Analysis ---\n")
        print(f"Assortativity Coefficient (r): {r:.4f}")
        if r > 0:
            print("Streamers tend to connect with others of the same label.")
        else:
            print("Connections are mixed; structural prediction might be hard.")
    except Exception as e:
        print(f"Could not calculate assortativity: {e}")

    # Load features
    print("\nLoading features...")

    with open(features_file, "rb") as f:
        features_dict = orjson.loads(f.read())

    # Extract nodes based on the 'mature' attribute
    mature_nodes = [n for n, data in G.nodes(data=True) if data.get("mature") is True]
    safe_nodes = [n for n, data in G.nodes(data=True) if data.get("mature") is False]

    print(
        f"Analyzing {len(mature_nodes)} mature nodes and {len(safe_nodes)} safe nodes"
    )

    top_mature = get_top_features(mature_nodes, features_dict)
    top_safe = get_top_features(safe_nodes, features_dict)

    # Display as tables
    if top_mature:

        df_mature = pl.DataFrame(
            {
                "Rank": range(1, len(top_mature) + 1),
                "Game_ID": [g for g, _ in top_mature],
                "Count": [c for _, c in top_mature],
            }
        )
        print("\nTop features (Games) for 'Mature' Streamers:")
        print(df_mature)

    if top_safe:
        df_safe = pl.DataFrame(
            {
                "Rank": range(1, len(top_safe) + 1),
                "Game_ID": [g for g, _ in top_safe],
                "Count": [c for _, c in top_safe],
            }
        )
        print("\nTop features (Games) for 'Non-Mature' Streamers:")
        print(df_safe)


def analyze_k_core(G):
    # Limpa self-loops (por segurança)
    G.remove_edges_from(nx.selfloop_edges(G))

    # Calcula o core number de cada nó
    core_numbers = nx.core_number(G)
    max_core = max(core_numbers.values())

    # Interpretação: O Max Core de 43 define o núcleo mais denso e coeso da rede, onde todos os nós
    # estão conectados a pelo menos 43 outros nós dentro desse subgrupo.
    print(f"   - Max Core (k-max): {max_core}")

    # Subgrafo do Max Core
    nodes_in_max_core = [n for n, k in core_numbers.items() if k == max_core]
    core_subgraph = G.subgraph(nodes_in_max_core)

    # Percentagem de 'Mature' no núcleo mais denso
    labels = nx.get_node_attributes(core_subgraph, "mature")
    mature_count = sum(labels.values())
    total_core = len(nodes_in_max_core)

    if total_core > 0:
        print(f"   - Tamanho do Max Core: {total_core} nós")
        # Interpretação: Uma percentagem elevada de 'Mature' no k-core máximo indicaria que os streamers de conteúdo maduro
        # dominam a estrutura central e mais resiliente da rede (o Core).
        print(
            f"   - Percentagem de 'Mature' no Core: {(mature_count/total_core)*100:.2f}%"
        )

    core_count = Counter(core_numbers.values())
    ks = sorted(core_count.keys())
    sizes = [sum(v for k2, v in core_count.items() if k2 >= k) for k in ks]

    plt.figure(figsize=(10, 6))
    plt.bar(ks, sizes, color="purple", alpha=0.7)
    plt.title("Tamanho acumulado de cada K-Core")
    plt.xlabel("k-core")
    plt.ylabel("Número de Nós ≥ k")
    plt.yscale("log")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(RESULTS_DIR, "k_core_sizes.png"))
    plt.close()


def simulate_diffusion(G):
    def find_hubs(G):
        """Encontra dinamicamente o hub Mature e Non-Mature com maior grau."""
        degrees = dict(G.degree())
        labels = nx.get_node_attributes(G, "mature")

        # Ordena os nós pelo Degree (maior primeiro)
        sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)

        # Encontra o hub Mature e o hub Non-Mature
        hub_mature = next((n for n in sorted_nodes if labels.get(n) == True), None)
        hub_non_mature = next((n for n in sorted_nodes if labels.get(n) == False), None)

        return hub_mature, hub_non_mature

    # Obtemos os Hubs conforme o output
    hub_mature, hub_non_mature = find_hubs(G)

    # Escolhe um nó aleatório para comparação
    random_node = 12345
    if random_node not in G:
        random_node = list(G.nodes())[0]

    degrees = dict(G.degree())

    print(f"   - Hub Mature: {hub_mature} (Grau: {degrees.get(hub_mature)})")
    # Interpretação: O Hub Non-Mature é o nó mais conectado de toda a rede (maior Degree Centrality).
    print(
        f"   - Hub Non-Mature: {hub_non_mature} (Grau: {degrees.get(hub_non_mature)})"
    )

    def run_si_model(start_node, steps=15, beta=0.1):
        """Modelo Suscetível-Infetado simples."""
        infected = {start_node}
        history = [1]
        total_nodes = G.number_of_nodes()

        for _ in range(steps):
            newly_infected = set()
            for node in infected:
                for neighbor in G.neighbors(node):
                    if neighbor not in infected:
                        if random.random() < beta:
                            newly_infected.add(neighbor)

            infected.update(newly_infected)
            history.append(len(infected))

            if len(infected) == total_nodes:
                break

        return history

    print("   A simular propagação...")
    steps = 20
    beta = 0.15

    hist_mature = run_si_model(hub_mature, steps=steps, beta=beta)
    hist_non_mature = run_si_model(hub_non_mature, steps=steps, beta=beta)
    hist_random = run_si_model(random_node, steps=steps, beta=beta)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        hist_mature,
        label=f"Início: Hub Mature (Grau {degrees[hub_mature]})",
        color="red",
        marker="o",
    )
    plt.plot(
        hist_non_mature,
        label=f"Início: Hub Non-Mature (Grau {degrees[hub_non_mature]})",
        color="blue",
        marker="x",
    )
    plt.plot(
        hist_random,
        label=f"Início: Nó Aleatório (Grau {degrees[random_node]})",
        color="green",
        marker="s",
    )

    plt.title(f"Velocidade de Difusão de Informação (Modelo SI, Beta={beta})")
    plt.xlabel("Passos de Tempo")
    plt.ylabel("Total de Nós Infetados (Alcance)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "difusao_informacao.png"))
    plt.close()

    # Resultado final
    final_m = hist_mature[-1]
    final_nm = hist_non_mature[-1]
    final_r = hist_random[-1]

    winner = max(
        [("Mature", final_m), ("Non-Mature", final_nm), ("Random", final_r)],
        key=lambda x: x[1],
    )[0]

    # Interpretação: As Influence Cascades iniciadas nos hubs tendem a ser globais (Global Cascades), atingindo a maioria da rede.
    # O nó aleatório tipicamente gera cascatas locais, demonstrando a importância da posição do nó inicial na Information Diffusion.
    print(
        f"    Alcance após {steps} passos: Mature={final_m}, Non-Mature={final_nm}, Random={final_r}"
    )
    print(f"   -> O conteúdo originado no hub '{winner}' espalhou-se mais rapidamente")


def menu():
    def run_pipeline(lang):
        edges_file = f"./data/{lang}/musae_{lang}_edges.csv"
        target_file = f"./data/{lang}/musae_{lang}_target.csv"
        feature_file = f"./data/{lang}/musae_{lang}_features.json"

        print(f"\n{lang} SELECTED\n")
        G = load_data(feature_file, target_file, edges_file)
        centrality_community(G)
        ploting_distribuitions(G)
        deep_analise(G, feature_file)
        simulate_diffusion(G)
        analyze_k_core(G)
        sys.stdout.flush()

    options = {
        1: "DE",
        2: "ENGB",
        3: "ES",
        4: "FR",
        5: "PTBR",
        6: "RU",
        7: "ALL",
        8: "ENGBDE",
        9: "EXIT",
    }

    try:
        op = int(
            input(
                "\n SELECT OPTION Number\n"
                "  1: DE\n"
                "  2: ENGB\n"
                "  3: ES\n"
                "  4: FR\n"
                "  5: PTBR\n"
                "  6: RU\n"
                "  7: ALL\n"
                "  8: ENGBDE\n"
                "  9: EXIT\n"
                " >>> "
            )
        )
    except ValueError:
        print("\nInvalid input\n")
        return

    choice = options.get(op)

    if choice is None:
        print("\nUnknown option\n")
        return

    if choice == "EXIT":
        print("\nBYE BYE!\n")
        sys.stdout.flush()
        sys.exit()

    run_pipeline(choice)


menu()
