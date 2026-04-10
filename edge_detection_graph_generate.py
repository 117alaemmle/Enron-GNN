import pandas as pd
import email
import torch

# 1. Load a sample of the data (Start small, the full CSV has ~500,000 rows!)
print("Loading data...")
df = pd.read_csv('./Kaggle_DataSet/emails.csv', nrows=None)

# 2. Extract Senders and Receivers using Python's built-in email parser
print("Parsing emails...")
edges_text = []

for raw_message in df['message']:
    # Parse the raw text into an email object
    msg = email.message_from_string(raw_message)
    
    sender = msg.get('From')
    receivers_raw = msg.get('To')
    
    # Clean up the data and handle multiple recipients
    if sender and receivers_raw:
        sender = sender.strip().lower()
        # Emails often have multiple recipients separated by commas and newlines
        receivers = [r.strip().lower() for r in receivers_raw.replace('\n', '').split(',')]
        
        for receiver in receivers:
            if receiver: # Make sure it's not empty
                edges_text.append((sender, receiver))

# 3. Create a unique ID mapping for every email address (Nodes)
print("Mapping emails to integer IDs...")
unique_emails = set([src for src, dst in edges_text] + [dst for src, dst in edges_text])

# This dictionary assigns a unique integer to every email address (e.g., {"jeff@enron.com": 0})
node_mapping = {email_addr: i for i, email_addr in enumerate(unique_emails)}

# 4. Convert text edges to integer edges
source_nodes = []
target_nodes = []

for sender, receiver in edges_text:
    source_nodes.append(node_mapping[sender])
    target_nodes.append(node_mapping[receiver])

# 5. Build the PyTorch Geometric edge_index tensor
# PyG expects a tensor of shape [2, num_edges] where row 0 is senders and row 1 is receivers
edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

print(f"\nSuccess! Found {len(unique_emails)} unique people (nodes).")
print(f"Found {len(source_nodes)} communication edges.")
print("Edge Index Shape:", edge_index.shape)
print("\nFirst 5 edges (Sender ID -> Receiver ID):")
print(edge_index[:, :5])



# 6. Visualize the graph using NetworkX

import networkx as nx
import matplotlib.pyplot as plt

print("Converting PyTorch edges to NetworkX...")
# 1. Convert the PyTorch tensor [2, num_edges] back to a list of (sender, receiver) pairs
edges_list = edge_index.t().tolist()

# 2. Create a Directed Graph (Emails go FROM someone TO someone)
G = nx.DiGraph()
G.add_edges_from(edges_list)

print(f"Full Graph -> Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# 3. The "Hairball" Fix: Find the most active nodes
# Count how many connections (in + out) each node has
node_degrees = dict(G.degree())

# Sort them and grab the IDs of the top 50 most connected people
top_50_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)[:50]

# Create a "subgraph" that only contains these 50 VIPs and the edges between them
sub_G = G.subgraph(top_50_nodes)

# # 4. Draw the Graph
# plt.figure(figsize=(12, 12))

# # The layout algorithm calculates where to place the dots. 
# # "Spring layout" treats edges like physical springs, pulling connected people together.
# pos = nx.spring_layout(sub_G, k=0.3, iterations=50)

# WE DON'T WANT TO PRINT THIS GRAPH WE ONLY WANT THE ONE WITH LABELS

# print("Drawing the top 50 communicators...")
# nx.draw(
#     sub_G, 
#     pos, 
#     node_size=[node_degrees[n] * 2 for n in sub_G.nodes()], # Make highly active nodes bigger
#     node_color='coral', 
#     edge_color='lightgray', 
#     alpha=0.8,
#     arrows=True, # Show the direction of the email
#     with_labels=False # We'll keep IDs off for now to keep it clean
# )

# plt.title("Enron Network: Top 50 Most Active Accounts", fontsize=16)
# # plt.savefig("enron_graph.png") # Uncomment this to save the image to your folder
# plt.show()


# Add Labels to the Graph

import networkx as nx
import matplotlib.pyplot as plt

print("Decoding IDs back to names...")
# 1. Reverse our original dictionary (Turns { "jeff@enron.com": 0 } into { 0: "jeff@enron.com" })
id_to_email = {v: k for k, v in node_mapping.items()}

# 2. Create a clean label dictionary for just our top 50 nodes
labels = {}
for n in sub_G.nodes():
    full_email = id_to_email[n]
    # Strip the domain to make the plot readable (e.g., "jeff.skilling" instead of the full email)
    clean_name = full_email.replace("@enron.com", "").split('@')[0] 
    labels[n] = clean_name

# 3. Set up a larger canvas for the text
plt.figure(figsize=(16, 16)) 

# Recalculate the layout (increasing 'k' pushes the nodes a bit further apart so text doesn't overlap)
pos = nx.spring_layout(sub_G, k=0.5, iterations=50)

print("Drawing the labeled graph...")
# Draw Nodes
nx.draw_networkx_nodes(
    sub_G, 
    pos, 
    node_size=[node_degrees[n] * 2 for n in sub_G.nodes()], 
    node_color='skyblue', 
    alpha=0.7
)

# Draw Edges
nx.draw_networkx_edges(
    sub_G, 
    pos, 
    edge_color='lightgray', 
    alpha=0.5, 
    arrows=True
)

# Draw Labels!
nx.draw_networkx_labels(
    sub_G, 
    pos, 
    labels=labels, 
    font_size=9, 
    font_weight="bold",
    font_family="sans-serif"
)

plt.title("Enron VIPs: Who was talking to whom?", fontsize=18)
plt.axis("off") # Turns off the grid/box around the plot
plt.show()


#Create the PyTorch Geometric Graph


import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree
import torch_geometric.transforms as T

print("1. Building Node Features (X)...")
num_nodes = len(unique_emails)

# Calculate Out-Degree (Emails Sent) from the first row of edge_index
out_degree = degree(edge_index[0], num_nodes=num_nodes)

# Calculate In-Degree (Emails Received) from the second row of edge_index
in_degree = degree(edge_index[1], num_nodes=num_nodes)

# Combine them into a single feature matrix X 
# Shape will be [num_nodes, 2] -> Every person gets a 2-number profile
x = torch.stack([out_degree, in_degree], dim=1)

print(f"Feature matrix X shape: {x.shape}")
print(f"Profile for Node 0 (Sent, Received): {x[0].tolist()}")

print("\n2. Creating the Master Data Object...")
# Bundle the features and the map together into one PyG Data object
data = Data(x=x, edge_index=edge_index)
print(data)

print("\n3. Splitting graph into Train/Validation/Test sets...")
# This transform magically handles our positive and negative edge sampling!
transform = T.RandomLinkSplit(
    num_val=0.1,   # 10% of edges used to tune the model
    num_test=0.2,  # 20% of edges hidden for the final exam
    is_undirected=False, # Emails are directed (A -> B is different than B -> A)
    add_negative_train_samples=True # Generates "fake" emails to teach it what a 0 is
)

train_data, val_data, test_data = transform(data)

print("\nSuccess! Here is your Training Graph:")
print(train_data)
print("\nNotice that train_data now has 'edge_label' and 'edge_label_index'.")
print("These are the answers (1 for real emails, 0 for fake emails) your model will learn from.")

# --- Add this to the end of your current script ---

print("\n4. Saving graph data to disk...")
# We pack the three datasets into a Python dictionary and save it as a .pt file
torch.save({
    'train': train_data,
    'val': val_data,
    'test': test_data
}, 'enron_graph_data.pt')

print("Success! Data saved to 'enron_graph_data.pt'. You can now close this script.")