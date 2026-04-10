# motif_validation.py
import torch
from motif_detection_train import MotifGNN # Import your model class

# 1. Load the data and the trained brain
data = torch.load('motif_enron_data.pt', weights_only=False)
model = MotifGNN(data.x.shape[1], 32, 6)
model.load_state_dict(torch.load('motif_model_weights.pth'))
model.eval()

# 2. Get predictions
with torch.no_grad():
    logits = model(data.x.float(), data.edge_index)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

# 3. Check the "Unlabeled" nodes (The ones the model had to guess)
unlabeled_mask = ~data.train_mask
print(f"Checking {unlabeled_mask.sum()} unlabeled nodes...")

# Let's see what it thinks of the general population
for i in range(20): # Look at the first 20 unknown nodes
    if unlabeled_mask[i]:
        print(f"Node {i}: Predicted Role {preds[i].item()} (Confidence: {probs[i][preds[i]].item():.2f})")