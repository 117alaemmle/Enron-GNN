# This script processes the Enron email dataset to extract advanced structural features for each email address (node) in the communication network. It constructs a directed graph using NetworkX, computes various topological features such as in-degree, out-degree, PageRank, clustering coefficient, and HITS scores (hubs and authorities). Finally, it packages these features into a PyTorch Geometric Data object for use in graph neural network models.

import pandas as pd
import email
import torch
import networkx as nx
from torch_geometric.data import Data
import numpy as np
from sklearn.preprocessing import StandardScaler

print("1. Loading Enron Corpus...")
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

# 2. BUILD THE RAW GRAPH
print("2. Building the NetworkX Directed Graph...")
G_raw = nx.DiGraph()
G_raw.add_edges_from(edges_text)
G_raw.remove_edges_from(nx.selfloop_edges(G_raw))

# --- NEW TWEAK #1: THE K-CORE FILTER ---
# Drop anyone who hasn't participated in at least 10 emails
print("   -> Filtering out the 'Ghosts' (Degree < 10)...")
core_nodes = [n for n, d in G_raw.degree() if d >= 10]
G = G_raw.subgraph(core_nodes).copy()

unique_emails = list(G.nodes())
node_mapping = {email_addr: i for i, email_addr in enumerate(unique_emails)}
num_nodes = len(unique_emails)

print(f"   -> Graph reduced from {len(G_raw.nodes())} to an active core of {num_nodes} nodes.")

# 3. CALCULATE ADVANCED TOPOLOGICAL FEATURES
print("3. Calculating Advanced Network Geometry...")

in_degrees = dict(G.in_degree())
out_degrees = dict(G.out_degree())

print("   -> Computing PageRank...")
pagerank = nx.pagerank(G, alpha=0.85)

print("   -> Computing Clustering Coefficients...")
clustering = nx.clustering(G.to_undirected())

print("   -> Computing Hubs and Authorities...")
try:
    hubs, authorities = nx.hits(G, max_iter=100, normalized=True)
except nx.PowerIterationFailedConvergence:
    print("   [!] HITS failed to converge. Defaulting to 0.")
    hubs = {node: 0.0 for node in G.nodes()}
    authorities = {node: 0.0 for node in G.nodes()}

# 4. CONSTRUCT AND SCALE THE FEATURE TENSOR (X)
print("4. Normalizing and Scaling Features...")
feature_matrix = []

for node in unique_emails:
    f_in = np.log1p(in_degrees[node])
    f_out = np.log1p(out_degrees[node])
    f_pr = pagerank[node]
    f_clust = clustering[node]
    f_hub = hubs[node]
    f_auth = authorities[node]
    
    feature_matrix.append([f_in, f_out, f_pr, f_clust, f_hub, f_auth])

# --- NEW TWEAK #2: STANDARD SCALER ---
# This forces every single metric to have a mean of 0 and a standard deviation of 1.
# Now, PageRank and Clustering have equal "voting power" in determining the shape!
scaler = StandardScaler()
scaled_matrix = scaler.fit_transform(feature_matrix)

x = torch.tensor(scaled_matrix, dtype=torch.float)

# 5. CONSTRUCT PYTORCH GEOMETRIC DATA
print("5. Packaging Data for Graph Neural Network...")
source_nodes = [node_mapping[src] for src, dst in G.edges()]
target_nodes = [node_mapping[dst] for src, dst in G.edges()]
edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

data = Data(x=x, edge_index=edge_index)
data.email_mapping = {i: em for em, i in node_mapping.items()}

torch.save(data, 'archetype_enron_data.pt')
print(f"Success! Saved scaled geometric profiles to 'archetype_enron_data.pt'.")