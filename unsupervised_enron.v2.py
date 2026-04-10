import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
import pandas as pd
import email
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# 1. LOAD DATA & MODEL (Keeping your existing logic)
print("Loading graph data...")
data = torch.load('enron_node_class_data.pt', weights_only=False)

# Re-mapping IDs to Emails for the hover labels
# (Note: Using the same logic from our build script)
df = pd.read_csv('./Kaggle_DataSet/emails.csv', nrows=50000)
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

unique_emails = sorted(list(set([src for src, dst in edges_text] + [dst for src, dst in edges_text])))
id_to_email = {i: email_addr for i, email_addr in enumerate(unique_emails)}

# 2. ENCODER DEFINITION
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

model = GAE(GCNEncoder(data.x.shape[1], 16))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 3. TRAINING (Simplified for quick check)
print("Training...")
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x.float(), data.edge_index)
    loss = model.recon_loss(z, data.edge_index)
    loss.backward()
    optimizer.step()

# 4. DIMENSIONALITY REDUCTION & CLUSTERING
model.eval()
with torch.no_grad():
    z = model.encode(data.x.float(), data.edge_index)
    z_np = z.cpu().numpy()

print("Clustering and reducing for Plotly...")
kmeans = KMeans(n_clusters=8, n_init=10).fit(z_np)
tsne = TSNE(n_components=2, perplexity=30)
z_2d = tsne.fit_transform(z_np)

# 5. CREATE INTERACTIVE PLOT
# Create a DataFrame for Plotly
viz_df = pd.DataFrame({
    'x': z_2d[:, 0],
    'y': z_2d[:, 1],
    'email': [id_to_email[i] for i in range(len(unique_emails))],
    'cluster': kmeans.labels_.astype(str)
})

fig = px.scatter(
    viz_df, x='x', y='y', 
    color='cluster', 
    hover_name='email',
    title="Enron Unsupervised Social Map (Hover to identify employees)",
    template="plotly_dark"
)

# This will open a tab in your default web browser
fig.show()