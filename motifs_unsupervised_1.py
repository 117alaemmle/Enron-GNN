# This script processes the Enron email dataset to extract advanced structural features for each email address (node) in the communication network. It constructs a directed graph using NetworkX, computes various topological features such as in-degree, out-degree, PageRank, clustering coefficient, and HITS scores (hubs and authorities). Finally, it packages these features into a PyTorch Geometric Data object for use in graph neural network models.

import pandas as pd
import email
import torch
import networkx as nx
from torch_geometric.data import Data
import numpy as np

print("1. Loading Enron Corpus...")
# Load a healthy chunk. 500k is great if your RAM can handle the NetworkX math!
df = pd.read_csv('./Kaggle_DataSet/emails.csv', nrows=500000)

edges_text = []
for raw_message in df['message']:
    msg = email.message_from_string(raw_message)
    sender = msg.get('From')
    receivers_raw = msg.get('To')
    
    if sender and receivers_raw:
        sender = sender.strip().lower()
        receivers = [r.strip().lower() for r in receivers_raw.replace('\n', '').split(',')]
        for r in receivers:
            if r: edges_text.append((sender, r))

# 2. BUILD THE NETWORKX GRAPH
print("2. Building the NetworkX Directed Graph...")
G = nx.DiGraph()
G.add_edges_from(edges_text)

# Remove self-loops (people emailing themselves) as it skews the math
G.remove_edges_from(nx.selfloop_edges(G))

# Map nodes to integer IDs
unique_emails = list(G.nodes())
node_mapping = {email_addr: i for i, email_addr in enumerate(unique_emails)}
num_nodes = len(unique_emails)

print(f"Graph initialized with {num_nodes} unique email addresses.")

# 3. CALCULATE ADVANCED TOPOLOGICAL FEATURES
print("3. Calculating Advanced Network Geometry (This may take a minute)...")

# Base Volume
in_degrees = dict(G.in_degree())
out_degrees = dict(G.out_degree())

# PageRank (The VIP Metric)
print("   -> Computing PageRank...")
pagerank = nx.pagerank(G, alpha=0.85)

# Clustering Coefficient (The Clique Metric)
print("   -> Computing Clustering Coefficients...")
# Clustering is mathematically intense on directed graphs, so we convert to undirected for this specific metric
clustering = nx.clustering(G.to_undirected())

# HITS Algorithm (Hubs and Authorities)
print("   -> Computing Hubs and Authorities...")
try:
    hubs, authorities = nx.hits(G, max_iter=100, normalized=True)
except nx.PowerIterationFailedConvergence:
    print("   [!] HITS failed to converge. Defaulting to 0.")
    hubs = {node: 0.0 for node in G.nodes()}
    authorities = {node: 0.0 for node in G.nodes()}

# 4. CONSTRUCT THE FEATURE TENSOR (X)
print("4. Normalizing and Stacking Features...")
feature_matrix = []

for node in unique_emails:
    # We use log1p (log(1 + x)) for volume metrics because they follow a power-law distribution
    # We multiply probability metrics (like PageRank) by large scalars to prevent vanishing gradients
    
    f_in = np.log1p(in_degrees[node])
    f_out = np.log1p(out_degrees[node])
    f_pr = pagerank[node] * 1000       # Scale up
    f_clust = clustering[node]         # Already 0 to 1
    f_hub = hubs[node] * 1000          # Scale up
    f_auth = authorities[node] * 1000  # Scale up
    
    feature_matrix.append([f_in, f_out, f_pr, f_clust, f_hub, f_auth])

x = torch.tensor(feature_matrix, dtype=torch.float)

# 5. CONSTRUCT PYTORCH GEOMETRIC DATA
print("5. Packaging Data for Graph Neural Network...")
source_nodes = [node_mapping[src] for src, dst in G.edges()]
target_nodes = [node_mapping[dst] for src, dst in G.edges()]
edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

# We save the mapping dictionary directly into the PyG Data object so we don't lose the names!
data = Data(x=x, edge_index=edge_index)
data.email_mapping = {i: em for em, i in node_mapping.items()}

torch.save(data, 'archetype_enron_data.pt')
print(f"Success! Saved full geometric profiles to 'archetype_enron_data.pt'.")