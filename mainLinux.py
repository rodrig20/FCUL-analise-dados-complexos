r"""
Luis Palma: 58842
Rodrigo Lopes:
Diogo :

Create virtual enviorment:  py -3.10 -m venv venv
Activate VENV  .\venv\Scripts\Activate
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


r"""
Exemplo de PTBR
features_file: data\PTBR\musae_PTBR_features.json
target_file:data\PTBR\musae_PTBR_target.csv
eadges file: data\PTBR\musae_PTBR_edges.csv
"""

""" Function for creating graph using the sample language like PTBR, DE, FR , RU, ENGB, ES
requires: Three files
    features_file: feature file in json format
    target_files: target file in csv format
    eadges_file: degree in csv format
"""


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


def centrality_community(g):

    # --- 1. Centrality Features ---
    print("\n--- Centrality Features (Top 5) ---")

    # Calculate all centralities
    degree_cent = nx.degree_centrality(g)
    betweenness_cent = nx.betweenness_centrality(g)
    closeness_cent = nx.closeness_centrality(g)

    # get top 5
    def get_top_5(cent_dict):
        return sorted(cent_dict.items(), key=lambda x: x[1], reverse=True)[:5]

    top5_degree = get_top_5(degree_cent)
    top5_betweenness = get_top_5(betweenness_cent)
    top5_closeness = get_top_5(closeness_cent)

    # Create a nice DataFrame for display
    df_top5 = pl.DataFrame(
        {
            "Rank": [1, 2, 3, 4, 5],
            "Degree (Node: Score)": [f"{n}: {s:.4f}" for n, s in top5_degree],
            "Betweenness (Node: Score)": [f"{n}: {s:.4f}" for n, s in top5_betweenness],
            "Closeness (Node: Score)": [f"{n}: {s:.4f}" for n, s in top5_closeness],
        }
    )

    # Print the table without the index number
    print(df_top5)

    # --- 2. Community Detection ---
    # Create a nice DataFrame for display
    print("\n--- Community Detection (Top 5 by Size) ---")

    # Run Algorithms and convert to list
    try:
        communities_gmc = list(nx.community.greedy_modularity_communities(g))
    except AttributeError:
        communities_gmc = []

    try:
        communities_louvain = list(nx.community.louvain_communities(g))
    except AttributeError:
        communities_louvain = []  # Fallback if method missing or library too old

    # Lambda: Sorts communities by SIZE (len), returns (Index, Size)
    get_top_5_comms = lambda comms: sorted(
        [(i, len(c)) for i, c in enumerate(comms)], key=lambda x: x[1], reverse=True
    )[:5]

    top5_gmc = get_top_5_comms(communities_gmc)
    top5_louvain = get_top_5_comms(communities_louvain)

    # Pad with placeholders if < 5 found (just for table safety)
    # while len(top5_gmc) < 5: top5_gmc.append(("-", 0))
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

    # Small World Properties ---
    print("\n--- Small World Properties ---")

    if nx.is_connected(g):
        G_lcc = g
    else:
        # Use Largest Connected Component
        G_lcc = g.subgraph(max(nx.connected_components(g), key=len))

    avg_clustering = nx.average_clustering(g)
    avg_path_length = nx.average_shortest_path_length(G_lcc)
    if len(G_lcc) > 1 and nx.is_connected(G_lcc):
        avg_path_length = nx.average_shortest_path_length(G_lcc)
    else:
        # Handle case where LCC is too small or not connected
        avg_path_length = float("nan")

    # Sigma (Small-Worldness)

    if not math.isnan(avg_path_length) and avg_path_length != 0:
        sigma_simple = avg_clustering / avg_path_length
    else:
        sigma_simple = float("nan")

    data = {
        "Property": [
            "Avg Clustering Coefficient ",
            f"Avg Path Length ",
            "Simple Ratio ",
        ],
        "Value": [
            f"{avg_clustering:.4f}",
            (
                f"{avg_path_length:.4f}"
                if not math.isnan(avg_path_length)
                else "N/A (LCC too small)"
            ),
            f"{sigma_simple:.4f}" if not math.isnan(sigma_simple) else "N/A",
        ],
    }

    df_sw = pl.DataFrame(data)

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
    degrees = [d for n, d in G.degree()]
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


"""
eadges_file = "data\PTBR\musae_PTBR_edges.csv"
target_file= "data\PTBR\musae_PTBR_target.csv"
feature_file= "data\PTBR\musae_PTBR_features.json"

G= load_data(feature_file, target_file, eadges_file)
centrality_community(G)
ploting_distribuitions(G)

"""
"""
this function Analysis of features
Shows how the Homophily Analysis
shows the top 5 games for each type streamer "mature" and "Non-Mature"



"""


def deep_analise(G, features_file):
    # FIXED: Simplified access logic
    def get_top_features(node_list, features_dict, top_n=5):
        games = []
        for n in node_list:
            # DIRECT ACCESS: The value is already the list of games
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


def menu():
    op = int(
        input(
            "\n SELECT OPTION Number  \n\
        1: DE \n\
        2: ENGB\n\
        3: ES \n\
        4: FR \n\
        5: PTBR \n\
        6: RU \n\
        7: ALL\n\
        8: EXIT\n "
        )
    )

    def DE():
        print("\nDE SELECTED\n")
        eadges_file = "data/DE/musae_DE_edges.csv"
        target_file = "data/DE/musae_DE_target.csv"
        feature_file = "data/DE/musae_DE_features.json"
        G = load_data(feature_file, target_file, eadges_file)
        centrality_community(G)
        ploting_distribuitions(G)
        deep_analise(G, feature_file)
        sys.stdout.flush()

        return "DE selected"

    def ENGB():
        print("\nENGB SELECTED \n")
        eadges_file = "data/ENGB/musae_ENGB_edges.csv"
        target_file = "data/ENGB/musae_ENGB_target.csv"
        feature_file = "data/ENGB/musae_ENGB_features.json"
        G = load_data(feature_file, target_file, eadges_file)
        centrality_community(G)
        ploting_distribuitions(G)
        deep_analise(G, feature_file)
        sys.stdout.flush()

        return

    def ES():
        print("\n ES SELECTED \n")
        eadges_file = "data/ES/musae_ES_edges.csv"
        target_file = "data/ES/musae_ES_target.csv"
        feature_file = "data/ES/musae_ES_features.json"
        G = load_data(feature_file, target_file, eadges_file)
        centrality_community(G)
        ploting_distribuitions(G)
        deep_analise(G, feature_file)

        sys.stdout.flush()
        return

    def FR():
        print("\nFR SELECTED\n ")
        eadges_file = "data/FR/musae_FR_edges.csv"
        target_file = "data/FR/musae_FR_target.csv"
        feature_file = "data/FR/musae_FR_features.json"
        G = load_data(feature_file, target_file, eadges_file)
        centrality_community(G)
        ploting_distribuitions(G)
        deep_analise(G, feature_file)
        sys.stdout.flush()
        return

    def PTBR():
        print("\nPTBR SELECTED\n ")
        eadges_file = "data/PTBR/musae_PTBR_edges.csv"
        target_file = "data/PTBR/musae_PTBR_target.csv"
        feature_file = "data/PTBR/musae_PTBR_features.json"

        G = load_data(feature_file, target_file, eadges_file)
        centrality_community(G)
        ploting_distribuitions(G)
        deep_analise(G, feature_file)
        sys.stdout.flush()

        return

    def RU():
        print("\nRU SELECTED \n")
        eadges_file = "data/RU/musae_RU_edges.csv"
        target_file = "data/RU/musae_RU_target.csv"
        feature_file = "data/RU/musae_RU_features.json"
        G = load_data(feature_file, target_file, eadges_file)
        centrality_community(G)
        ploting_distribuitions(G)
        deep_analise(G, feature_file)
        sys.stdout.flush()
        return

    def ALL():
        print("\nALL SELECTED\n")

        eadges_file = "data/ALL/musae_ALL_edges.csv"
        target_file = "data/ALL/musae_ALL_target.csv"
        feature_file = "data/ALL/musae_ALL_features.json"
        G = load_data(feature_file, target_file, eadges_file)
        centrality_community(G)
        ploting_distribuitions(G)
        deep_analise(G, feature_file)
        sys.stdout.flush()

        return

    def EXIT():
        print(" BYE BYE, have a GREAT DAY!!!")
        sys.stdout.flush()
        sys.exit()

    switcher = {1: DE, 2: ENGB, 3: ES, 4: FR, 5: PTBR, 6: RU, 7: ALL, 8: EXIT}

    value = op
    result = switcher.get(value, lambda: "unknown")()
    print(result)


menu()


"""
Windows

eadges_file = "data\PTBR\musae_PTBR_edges.csv"
target_file= "data\PTBR\musae_PTBR_target.csv"
feature_file= "data\PTBR\musae_PTBR_features.json"

G= load_data(feature_file, target_file, eadges_file)
centrality_community(G)
ploting_distribuitions(G)

"""


"""
Linux

eadges_file = "data/PTBR/musae_PTBR_edges.csv"
target_file= "data/PTBR/musae_PTBR_target.csv"
feature_file= "data/PTBR/musae_PTBR_features.json"

G= load_data(feature_file, target_file, eadges_file)
deep_analise(G, feature_file)

"""
