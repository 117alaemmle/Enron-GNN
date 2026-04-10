import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
import os, io
import glob
from PIL import Image

# 1. SETUP THE DIRECTORY
slice_files = sorted(glob.glob('time_slices/graph_*.pt'))
if not slice_files:
    print("No time slices found! Run the Time Slicer script first.")
    exit()

# 2. DEFINE THE ENCODER
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

# 3. THE EVOLUTION LOOP
# We will look at a few key months to see the change
frames = []
for file_path in slice_files:
    month_label = file_path.split('_')[-1].replace('.pt', '')
    print(f"Processing Month: {month_label}...")
    
    data = torch.load(file_path, weights_only=False)
    
    # Initialize and train a fresh model for this month's "snapshot"
    model = GAE(GCNEncoder(data.x.shape[1], 8))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Short training burst to find the structure of THIS month
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x.float(), data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
    
    # Reduce to 2D
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x.float(), data.edge_index)
        z_2d = TSNE(n_components=2, perplexity=min(30, z.shape[0]-1)).fit_transform(z.cpu().numpy())
    
    # Create the Plot
    viz_df = pd.DataFrame({'x': z_2d[:, 0], 'y': z_2d[:, 1]})
    fig = px.scatter(viz_df, x='x', y='y', 
                     title=f"Enron Social Structure: {month_label}",
                     template="plotly_dark",
                     range_x=[-100, 100], range_y=[-100, 100]) # Keep scale consistent
    
    # Convert Plotly fig to raw PNG bytes in memory
    img_bytes = fig.to_image(format="png", width=800, height=600)
    
    # Open the bytes as a Pillow Image and add it to our frames list
    frames.append(Image.open(io.BytesIO(img_bytes)))
    print(f"   [+] Captured frame for {month_label}")
    
    # In a real scenario, we'd save these as PNGs to make a GIF
    # # For now, let's just show the most interesting months
    # if month_label in ['2000-01', '2001-01', '2001-10']:
    #     fig.show()

# Save all frames as a single looping GIF
if frames:
    print("\nStitching frames into GIF...")
    frames[0].save(
        "enron_evolution.gif",
        save_all=True,
        append_images=frames[1:],
        duration=500,  # 500 milliseconds per frame (half a second)
        loop=0         # 0 means loop infinitely
    )
    print("Success! Saved as 'enron_evolution.gif'.")