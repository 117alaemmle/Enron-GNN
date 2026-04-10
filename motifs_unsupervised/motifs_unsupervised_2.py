import torch
import numpy as np
from torch_geometric.nn import GCNConv, GAE
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import warnings

# Suppress K-Means warnings
warnings.filterwarnings('ignore')

print("1. Loading the Topological Data...")
data = torch.load('archetype_enron_data.pt', weights_only=False)
email_mapping = data.email_mapping

# A cheat-sheet of known VIPs to help us identify the clusters
known_vips = {
    # Executives
    "kenneth.lay": "Executive", "jeff.skilling": "Executive", "greg.whalley": "Executive", "louise.kitchen": "Executive",
    # Managers / Desk Heads
    "tim.belden": "Manager", "vince.kaminski": "Manager", "michael.kopper": "Manager", "mike.mcconnell": "Manager",
    # Traders / Specialists
    "john.arnold": "Trader", "james.derrick": "Legal/Specialist", "mark.haedicke": "Legal/Specialist",
    # Staff / Analysts
    "stinson.gibner": "Staff", "kevin.jordan": "Staff", "wanda.curry": "Staff",
    # Admins
    "rosalee.fleming": "Admin", "sherri.sera": "Admin", "nancy.mcneil": "Admin"
}

# 2. DEFINE THE UNSUPERVISED MODEL
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
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
    z = model.encode(data.x.float(), data.edge_index)
    loss = model.recon_loss(z, data.edge_index)
    loss.backward()
    optimizer.step()

print("\n3. Clustering the Network and Finding Archetypes...")
model.eval()
with torch.no_grad():
    z = model.encode(data.x.float(), data.edge_index).cpu().numpy()

num_roles = 6
kmeans = KMeans(n_clusters=num_roles, n_init=10, random_state=42)
clusters = kmeans.fit_predict(z)
centers = kmeans.cluster_centers_

print("\n" + "="*70)
print(" 🕵️‍♂️ CORPORATE ARCHETYPES & ROSETTA STONE")
print("="*70)

for i in range(num_roles):
    cluster_indices = np.where(clusters == i)[0]
    if len(cluster_indices) == 0: continue
    
    cluster_points = z[cluster_indices]
    centroid = centers[i].reshape(1, -1)
    distances = pairwise_distances(cluster_points, centroid).flatten()
    closest_idx_in_cluster = np.argsort(distances)
    
    print(f"\n🟢 CLUSTER {i} (Total Nodes: {len(cluster_indices)})")
    
    # --- PART A: THE UNKNOWN ARCHETYPES (The pure math) ---
    print(f"   [Top 10 Mathematical Archetypes]")
    found_internals = 0
    for idx in closest_idx_in_cluster:
        real_node_id = cluster_indices[idx]
        email_addr = email_mapping[real_node_id]
        
        if "@enron.com" in email_addr:
            print(f"     - {email_addr: <30} (Dist: {distances[idx]:.4f})")
            found_internals += 1
            
        if found_internals >= 10: break
            
    if found_internals == 0:
        print("     [!] Entirely External/Noise nodes.")

    # --- PART B: THE ROSETTA STONE (Checking for our knowns) ---
    print(f"   [Known VIPs Found in this Cluster]")
    vips_found = 0
    for idx in range(len(cluster_indices)):
        real_node_id = cluster_indices[idx]
        email_addr = email_mapping[real_node_id]
        prefix = email_addr.split('@')[0]
        
        if prefix in known_vips:
            # We found one of our known people!
            print(f"     ⭐ {email_addr: <28} -> Ground Truth: {known_vips[prefix]}")
            vips_found += 1
            
    if vips_found == 0:
        print("     (No known VIPs landed in this cluster)")