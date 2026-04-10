# RUN AFTER motif_detection_build TO TRAIN THE NEURAL NETWORK ON MOTIF-DEPENDENT GRAPH.
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv # GraphSAGE is excellent for motif patterns

# 1. LOAD DATA
data = torch.load('motif_enron_data.pt', weights_only=False)

# 2. DEFINE THE GNN (GraphSAGE)
class MotifGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# 6 classes: 0:Exec, 1:Manager, 2:Specialist, 3:Staff, 4:Admin, 5:External
model = MotifGNN(data.x.shape[1], 32, 6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 3. APPLY CLASS WEIGHTS
# We give much higher weight to Admins (4) and Executives (0) 
# because they are rare but have very distinct 'shapes' in the graph.
weights = torch.tensor([4.0, 2.0, 2.0, 1.0, 8.0, 0.5]) 
criterion = torch.nn.CrossEntropyLoss(weight=weights)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x.float(), data.edge_index)
    # Only calculate loss on nodes we actually labeled in the build script
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

print("Training Motif Detector...")
for epoch in range(1, 301):
    loss = train()
    if epoch % 50 == 0:
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')

# 4. SAVE THE TRAINED BRAIN
torch.save(model.state_dict(), 'motif_model_weights.pth')
print("\nSuccess! Model weights saved to 'motif_model_weights.pth'")