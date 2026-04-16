import pandas as pd
import email
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

print("1. Loading Enron Corpus...")
SCRIPT_DIR = Path(__file__).parent
file_path = SCRIPT_DIR.parent / 'Kaggle_DataSet' / 'emails.csv'

if not file_path.exists():
    file_path = SCRIPT_DIR / 'Kaggle_DataSet' / 'emails.csv'
    if not file_path.exists():
        file_path = '../Kaggle_DataSet/emails.csv'

df = pd.read_csv(file_path)

print("2. Building the Network...")
G_raw = nx.Graph() 

edges_to_add = []
for raw_message in df['message']:
    msg = email.message_from_string(raw_message)
    sender = msg.get('From')
    receivers_raw = msg.get('To')
    
    if sender and receivers_raw:
        sender = sender.strip().lower()
        receivers = [r.strip().lower() for r in receivers_raw.replace('\n', '').split(',')]
        for r in receivers:
            if r: 
                edges_to_add.append((sender, r))

G_raw.add_edges_from(edges_to_add)
G_raw.remove_edges_from(nx.selfloop_edges(G_raw))

print("3. Pruning the 'Hairball' (Filtering noise)...")
# Filter: Keep nodes with at least 25 connections
core_nodes = [n for n, d in G_raw.degree() if d >= 25]
G_core = G_raw.subgraph(core_nodes).copy()

print("4. Removing disconnected outliers...")
largest_cc = max(nx.connected_components(G_core), key=len)
G = G_core.subgraph(largest_cc).copy()

print(f"   Final Graph for Plotting: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

print("5. Calculating Layout and Node Sizes...")
degrees = dict(G.degree())
node_sizes = [v * 3 for v in degrees.values()] 

node_colors = []
for node in G.nodes():
    if "@enron.com" in node:
        node_colors.append('#4C72B0') 
    else:
        node_colors.append('#C44E52') 

pos = nx.spring_layout(G, k=0.20, iterations=50, seed=42)

print("6. Launching Interactive Viewer...")
# Launching an interactive window
fig = plt.figure(figsize=(18, 18), facecolor='#ffffff')
ax = plt.gca()
ax.set_facecolor('#ffffff')

nx.draw_networkx_edges(
    G, pos, 
    alpha=0.08, 
    edge_color='#999999', 
    width=0.6
)

nx.draw_networkx_nodes(
    G, pos, 
    node_size=node_sizes, 
    node_color=node_colors, 
    alpha=0.9, 
    edgecolors='white', 
    linewidths=0.5
)

plt.title("The Core Enron Network Architecture", 
          fontsize=24, fontweight='bold', color='#333333', pad=20)

import matplotlib.patches as mpatches
blue_patch = mpatches.Patch(color='#4C72B0', label='Internal (@enron.com)')
red_patch = mpatches.Patch(color='#C44E52', label='External Contacts')
plt.legend(handles=[blue_patch, red_patch], loc='upper right', fontsize=14, frameon=True)

plt.axis('off')
plt.tight_layout()

# This opens the interactive window so you can zoom!
plt.show()