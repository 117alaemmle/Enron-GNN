#This code attempts to learn the departments of Enron, placing them into their departments depending on whom they speak to.

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
import pandas as pd
import email
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# ==========================================
# 1. LOAD DATA & DEFINE NAMES
# ==========================================
print("1. Loading graph data and mapping emails...")
data = torch.load('enron_node_class_data.pt', weights_only=False)

# Re-mapping IDs to Email Prefixes for the highlighter
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

# ==========================================
# 2. THE MODEL ARCHITECTURE
# ==========================================
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

# Initialize model (16 dimensions for the latent space)
model = GAE(GCNEncoder(data.x.shape[1], 16))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ==========================================
# 3. TRAINING THE UNSUPERVISED GAE
# ==========================================
print("2. Training the GAE (Unsupervised)...")
for epoch in range(1, 151):
    model.train()
    optimizer.zero_grad()
    # model.encode() creates the latent 'z' coordinates
    z = model.encode(data.x.float(), data.edge_index)
    loss = model.recon_loss(z, data.edge_index)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"   Epoch {epoch:03d}: Loss {loss:.4f}")

# ==========================================
# 4. GENERATING FINAL COORDINATES (Z)
# ==========================================
model.eval()
with torch.no_grad():
    # This is the 'model.encode' step you needed!
    z = model.encode(data.x.float(), data.edge_index)
    z_np = z.cpu().numpy()

print("3. Reducing dimensions and clustering...")
kmeans = KMeans(n_clusters=8, n_init=10).fit(z_np)
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000)
z_2d = tsne.fit_transform(z_np)

# ==========================================
# 5. HIGHLIGHTER LOGIC
# ==========================================
anchors = {
    "West Desk (Traders)": ["tim.belden", "john.forney", "jeff.richter", "robert.badeer"],
    "Quants (Research)": ["vince.kaminski", "shirley.crenshaw", "stinson.gibner"],
    "C-Suite (Executives)": ["kenneth.lay", "jeff.skilling", "greg.whalley", "andrew.fastow"],
    "Deal Makers (LJM/Finance)": ["ben.glisan", "michael.kopper", "rick.buy"],
    "Legal": ["james.derrick", "mark.haedicke", "steven.kean"]
}

def identify_node(email_addr):
    prefix = email_addr.split('@')[0]
    for category, names in anchors.items():
        if prefix in names:
            return category
    return "General Staff"

viz_df = pd.DataFrame({
    'x': z_2d[:, 0],
    'y': z_2d[:, 1],
    'email': [id_to_email[i] for i in range(len(unique_emails))],
    'Group': [identify_node(id_to_email[i]) for i in range(len(unique_emails))]
})

# Make the VIPs larger so they pop
viz_df['Size'] = viz_df['Group'].apply(lambda x: 20 if x != "General Staff" else 6)

# ==========================================
# 6. PLOTLY INTERACTIVE VISUAL
# ==========================================
fig = px.scatter(
    viz_df, x='x', y='y', 
    color='Group',
    size='Size',
    hover_name='email',
    color_discrete_map={
        "West Desk (Traders)": "#FFD700",  # Gold
        "Quants (Research)": "#00FFFF",    # Cyan
        "C-Suite (Executives)": "#FF00FF", # Magenta
        "Deal Makers (LJM/Finance)": "#7FFF00", # Chartreuse
        "Legal": "#FFA500",               # Orange
        "General Staff": "#444444"         # Grey
    },
    title="Enron Forensic Map: The Silos Revealed",
    template="plotly_dark"
)

fig.show()