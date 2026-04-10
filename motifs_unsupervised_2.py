import torch
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv, GAE
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import warnings

# Suppress K-Means warnings
warnings.filterwarnings('ignore')

print("1. Loading the Topological Data...")
data = torch.load('archetype_enron_data.pt', weights_only=False)
email_mapping = data.email_mapping

# 2. DEFINE THE UNSUPERVISED MODEL
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # We take the 6D structural features and map them to a 16D hidden layer, 
        # then compress down to an 8D representation space
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, 8)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

model = GAE(GCNEncoder(data.x.shape[1], 8))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("2. Training the Graph Autoencoder (Blindly learning shapes)...")
for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    # Notice we pass the raw features and let the GNN compress them
    z = model.encode(data.x.float(), data.edge_index)
    loss = model.recon_loss(z, data.edge_index)
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"   Epoch {epoch:03d} | Loss: {loss.item():.4f}")

print("\n3. Clustering the Network and Finding Archetypes...")
model.eval()
with torch.no_grad():
    # 'z' contains the final 8D coordinates for every single person
    z = model.encode(data.x.float(), data.edge_index).cpu().numpy()

# We look for 6 basic corporate structures (e.g., Exec, Manager, Trader, Staff, Admin, External)
num_roles = 6
kmeans = KMeans(n_clusters=num_roles, n_init=10, random_state=42)
clusters = kmeans.fit_predict(z)
centers = kmeans.cluster_centers_

print("\n" + "="*50)
print(" CORPORATE ARCHETYPES DISCOVERED")
print("="*50)

for i in range(num_roles):
    # Find all nodes assigned to this specific cluster
    cluster_indices = np.where(clusters == i)[0]
    if len(cluster_indices) == 0: continue
    
    # Calculate how far each person in this cluster is from the exact mathematical center
    # The closer to 0, the more "perfect" their structural motif is
    cluster_points = z[cluster_indices]
    centroid = centers[i].reshape(1, -1)
    distances = pairwise_distances(cluster_points, centroid).flatten()
    
    # Sort from closest (most pure) to furthest (least pure)
    closest_idx_in_cluster = np.argsort(distances)
    
    print(f"\n🟢 CLUSTER {i} (Total Nodes: {len(cluster_indices)})")
    print(f"   Strongest Internal Archetypes:")
    
    found_internals = 0
    for idx in closest_idx_in_cluster:
        real_node_id = cluster_indices[idx]
        email_addr = email_mapping[real_node_id]
        
        # We filter for actual Enron employees to name the roles
        if "@enron.com" in email_addr:
            print(f"     - {email_addr: <30} (Distance: {distances[idx]:.4f})")
            found_internals += 1
            
        if found_internals >= 10:
            break
            
    if found_internals == 0:
        print("     [!] This cluster is entirely External/Noise nodes.")