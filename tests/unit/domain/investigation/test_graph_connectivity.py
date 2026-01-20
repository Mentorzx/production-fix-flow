"""Investigation: Graph Connectivity and Difficulty Analysis.

Checks if validation triples are 'reachable' from training data.
If (h, r, t) is in Valid, but 'h' and 't' are in different connected components
in the Train graph, standard KGE models often struggle (cold start link prediction).
"""

import networkx as nx
import numpy as np
import pytest

from pff.infrastructure.hpo.trials.data_loader import load_preprocessed_from_postgres
from pff.shared import FileManager, logger

# Disable excessive logging
logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO")


@pytest.mark.asyncio
async def test_graph_connectivity_difficulty():
    fm = FileManager()
    try:
        # Load Raw (mapped) Data
        train_df, valid_df, info = load_preprocessed_from_postgres(
            fm, require_preprocessed=False, auto_populate_if_missing=True
        )
    except Exception as e:
        pytest.skip(f"Failed to load data: {e}")
        return

    print(f"\n[DATA] Train: {len(train_df)}, Valid: {len(valid_df)}")

    # Build NetworkX graph from Train (Undirected for component analysis)
    # We treat all relations as edges.
    print("[GRAPH] Building Training Graph...")
    G = nx.Graph()

    # Polars to list of tuples is fast
    edges = train_df.select(["s", "p", "o"]).to_numpy()

    # Add edges (s, o)
    # Use only s and o columns (indices 0 and 2)
    edge_list = [(int(r[0]), int(r[2])) for r in edges]
    G.add_edges_from(edge_list)

    num_components = nx.number_connected_components(G)
    largest_cc = len(max(nx.connected_components(G), key=len))

    print(f"[GRAPH] Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"[GRAPH] Connected Components: {num_components}")
    print(f"[GRAPH] Largest Component Size: {largest_cc} ({largest_cc / G.number_of_nodes():.1%})")

    # Analyze Validation Difficulty
    print("[ANALYSIS] Checking Validation Triples...")
    valid_edges = valid_df.select(["s", "p", "o"]).to_numpy()

    same_component_count = 0
    new_nodes_count = 0
    total_valid = len(valid_edges)

    # Check a sample if too large
    sample_size = min(total_valid, 5000)
    if sample_size == 0:
        pytest.skip("Validation set is empty, skipping connectivity analysis.")

    indices = np.random.choice(total_valid, sample_size, replace=False)

    for i in indices:
        row = valid_edges[i]
        s, o = int(row[0]), int(row[2])

        has_s = G.has_node(s)
        has_o = G.has_node(o)

        if not has_s or not has_o:
            new_nodes_count += 1
            continue

        if nx.has_path(G, s, o):
            same_component_count += 1
            # Optional: Check shortest path length
            # length = nx.shortest_path_length(G, s, o)

    print(f"\n[RESULTS] Based on sample of {sample_size} validation triples:")
    print(
        f"  - Both entities in Train: {sample_size - new_nodes_count} ({(sample_size - new_nodes_count) / sample_size:.1%})"
    )
    print(
        f"  - In SAME Component (Reachability): {same_component_count} ({same_component_count / sample_size:.1%})"
    )
    print(
        f"  - Across Components (Hard/Impossible?): {sample_size - same_component_count - new_nodes_count}"
    )

    # Warning Thresholds
    if same_component_count / sample_size < 0.5:
        print("\n[WARNING] Less than 50% of validation pairs are in the same connected component!")
        print("          This implies the model has to infer links between disjoint subgraphs.")
        print("          Standard KGE (structural) might fail. Content/Attributes needed.")
    else:
        print("\n[OK] Connectivity looks reasonable (>50% reachable).")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_graph_connectivity_difficulty())
