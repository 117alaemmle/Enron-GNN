import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# ==========================================
# 1. THE MODEL: GRAPH AUTOENCODER
# ==========================================
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Layer 1: Aggregates neighbor info
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        # Layer 2: Compresses into the "Latent Space"
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

# ==========================================
# 2. LOAD DATA
# ==========================================
print("Loading graph data...")
# We use the node_class_data because it already has our log-normalized X features
data = torch.load('enron_node_class_data.pt', weights_only=False)

# Initialize GAE with our Encoder
# out_channels=16 means each person is reduced to 16 unique "structural numbers"
out_channels = 16
model = GAE(GCNEncoder(data.x.shape[1], out_channels))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ==========================================
# 3. THE TRAINING LOOP (Unsupervised)
# ==========================================
def train():
    model.train()
    optimizer.zero_grad()
    # z is the "Latent Representation" (The hidden department coordinates)
    z = model.encode(data.x.float(), data.edge_index)
    # The loss is how well the model can "guess" if an edge exists using only z
    loss = model.recon_loss(z, data.edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()

print("Training Unsupervised Model...")
for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

# ==========================================
# 4. CLUSTERING & VISUALIZATION
# ==========================================
model.eval()
with torch.no_grad():
    z = model.encode(data.x.float(), data.edge_index)

# A. K-Means: Find 6 "Natural" Departments
print("\nClustering nodes into 6 departments...")
kmeans = KMeans(n_clusters=6, n_init=10).fit(z.cpu().numpy())
labels = kmeans.labels_

# B. T-SNE: Squash 16D math into a 2D map we can actually look at
print("Reducing dimensions for visualization...")
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000)
z_2d = tsne.fit_transform(z.cpu().numpy())

# C. Plot the Enron "Universe"
plt.figure(figsize=(10, 8))
scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='Set1', s=10, alpha=0.6)
plt.colorbar(scatter, label='Predicted Department ID')
plt.title("Unsupervised Clustering of Enron Social Network\n(Coordinates based on GAE Latent Space)")
plt.xlabel("TSNE Dimension 1")
plt.ylabel("TSNE Dimension 2")
plt.show()

print("\nSuccess! The plot shows how Enron employees naturally clump together.")