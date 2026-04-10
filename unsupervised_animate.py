import torch
import os
import glob
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from PIL import Image
import io

# 1. SETUP
slice_files = sorted(glob.glob('time_slices/graph_*.pt'))
frames = []
output_dir = "gif_frames"
os.makedirs(output_dir, exist_ok=True)

print(f"Starting animation sequence for {len(slice_files)} months...")

# 2. GENERATE FRAMES
for i, file_path in enumerate(slice_files):
    month_label = file_path.split('_')[-1].replace('.pt', '')
    data = torch.load(file_path, weights_only=False)
    
    # We use a smaller latent space (8) for speed during animation
    from torch_geometric.nn import GCNConv, GAE
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(data.x.shape[1], 16)
            self.conv2 = GCNConv(16, 8)
        def forward(self, x, edge_index):
            return self.conv2(self.conv1(x, edge_index).relu(), edge_index)

    model = GAE(Encoder())
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Quick 50-epoch training to capture this month's "vibe"
    for _ in range(50):
        model.train(); opt.zero_grad(); z = model.encode(data.x.float(), data.edge_index)
        loss = model.recon_loss(z, data.edge_index); loss.backward(); opt.step()
    
    # Reduce and Plot
    z_2d = TSNE(n_components=2, perplexity=min(30, z.shape[0]-1)).fit_transform(z.detach().numpy())
    fig = px.scatter(x=z_2d[:, 0], y=z_2d[:, 1], 
                     title=f"Enron Internal Structure: {month_label}",
                     template="plotly_dark", range_x=[-70, 70], range_y=[-70, 70])
    
    # Convert Plotly fig to Image object
    img_bytes = fig.to_image(format="png", width=800, height=600)
    frames.append(Image.open(io.BytesIO(img_bytes)))
    print(f"   [+] Frame {i+1}/{len(slice_files)}: {month_label} captured.")

# 3. SAVE AS GIF
if frames:
    frames[0].save(
        "enron_evolution.gif",
        save_all=True,
        append_images=frames[1:],
        duration=500, # 500ms per month
        loop=0
    )
    print("\nSuccess! 'enron_evolution.gif' has been created.")