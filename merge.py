import polars as pl
import orjson
import os
import networkx as nx 

# Configuration
LANGUAGES = ['DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU']
BASE_PATH = "data"  
OUTPUT_FOLDER = os.path.join(BASE_PATH, "ALL")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global Containers
combined_edges = []
combined_targets = []
combined_features = {}

node_id_offset = 0

print(f"Starting merge process for: {LANGUAGES}\n")

for lang in LANGUAGES:
    print(f"Processing {lang}...")
    
    edges_file = os.path.join(BASE_PATH, lang, f"musae_{lang}_edges.csv")
    target_file = os.path.join(BASE_PATH, lang, f"musae_{lang}_target.csv")
    features_file = os.path.join(BASE_PATH, lang, f"musae_{lang}_features.json")
    
    if not os.path.exists(edges_file):
        print(f"  [!] Warning: {edges_file} not found. Skipping.")
        continue

    # 1. Load Data
    df_edges = pl.read_csv(edges_file)
    df_target = pl.read_csv(target_file)
    
    with open(features_file, 'rb') as f:
        data_features = orjson.loads(f.read())

    # 2. Calculate Max ID (Polars syntax)
    # We use .max() on the column series
    max_from = df_edges['from'].max()
    max_to = df_edges['to'].max()
    max_target = df_target['new_id'].max()
    
    # Handle cases where a file might be empty or None
    max_id = max(max_from or 0, max_to or 0, max_target or 0)
    num_nodes = max_id + 1
    
    print(f"  Found {num_nodes} nodes. Applying offset: +{node_id_offset}")

    # 3. Apply Offset using .with_columns() (THE FIX)
    
    # -- Edges --
    df_edges = df_edges.with_columns([
        (pl.col("from") + node_id_offset).alias("from"),
        (pl.col("to") + node_id_offset).alias("to")
    ])

    # -- Targets --
    df_target = df_target.with_columns([
        (pl.col("new_id") + node_id_offset).alias("new_id"),
        pl.lit(lang).alias("lang") # pl.lit() is used to add a constant string column
    ])

    # -- Features (Standard Python Dict logic, no change needed) --
    for original_id_str, feature_list in data_features.items():
        original_id = int(original_id_str)
        new_id = original_id + node_id_offset
        combined_features[str(new_id)] = feature_list

    # 4. Append
    combined_edges.append(df_edges)
    combined_targets.append(df_target)
    
    node_id_offset += num_nodes
    print("  Done.")

print("\nCombining all dataframes...")

# 5. Concat (Polars concat does not take ignore_index)
final_edges_df = pl.concat(combined_edges)
final_target_df = pl.concat(combined_targets)

print(f"Total Nodes: {len(combined_features)}")
print(f"Total Edges: {final_edges_df.height}")

# 6. Save (Polars uses write_csv)
print(f"\nSaving to {OUTPUT_FOLDER}...")

final_edges_df.write_csv(os.path.join(OUTPUT_FOLDER, "musae_ALL_edges.csv"))
final_target_df.write_csv(os.path.join(OUTPUT_FOLDER, "musae_ALL_target.csv"))

with open(os.path.join(OUTPUT_FOLDER, "musae_ALL_features.json"), 'wb') as f:
    f.write(orjson.dumps(combined_features))
    
    


def export_gexf_from_all(all_edges_csv, all_target_csv, all_features_json, output_gexf):
    # Load edges
    df_edges = pl.read_csv(all_edges_csv)
    edge_list = list(zip(df_edges["from"].to_list(), df_edges["to"].to_list()))

    # Load targets
    df_target = pl.read_csv(all_target_csv)
    mature_map = dict(zip(df_target["new_id"].to_list(), df_target["mature"].to_list()))
    lang_map = dict(zip(df_target["new_id"].to_list(), df_target["lang"].to_list()))

    # Load features
    with open(all_features_json, "rb") as f:
        features_dict = orjson.loads(f.read())

    # Build graph
    G = nx.Graph()
    G.add_edges_from(edge_list)
    nx.set_node_attributes(G, mature_map, "mature")
    nx.set_node_attributes(G, lang_map, "lang")
    nx.set_node_attributes(G, features_dict, "features")

    # Export to GEXF
    nx.write_gexf(G, output_gexf)
    print(f"GEXF file saved to: {output_gexf}")

# Usage example after merging:
export_gexf_from_all(
    os.path.join(OUTPUT_FOLDER, "musae_ALL_edges.csv"),
    os.path.join(OUTPUT_FOLDER, "musae_ALL_target.csv"),
    os.path.join(OUTPUT_FOLDER, "musae_ALL_features.json"),
    os.path.join(OUTPUT_FOLDER, "musae_ALL_network.gexf")
)

print("SUCCESS: All datasets merged.")