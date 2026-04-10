#This code trains a GNN to predict the roles of employees at ENron based on our already-built graph.
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# ==========================================
# 1. DEFINE THE NEURAL NETWORK
# ==========================================
class NodeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        # Layer 1: Takes our 2 features (In/Out degree) and expands them
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # Layer 2: Compresses the hidden patterns down to our 4 job categories
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # Pass features through the first graph layer
        x = self.conv1(x, edge_index)
        x = F.relu(x) # Activation function
        x = F.dropout(x, p=0.0, training=self.training) # Prevent memorization
        
        # Pass through the final layer to get the 4 category scores
        x = self.conv2(x, edge_index)
        return x

# ==========================================
# 2. LOAD DATA & TRAIN
# ==========================================
if __name__ == "__main__":
    print("1. Loading Phase 2 graph data...")
    # Load the Data object directly (since we didn't save it as a dictionary this time)
    data = torch.load('enron_node_class_data.pt', weights_only=False)
    
    print("2. Initializing Model...")
    # in_channels = 2 (In/Out Degree), num_classes = 7 (Exec, Trader, Legal, Admin, Accountant, Vendor, Bot)
    model = NodeClassifier(in_channels=2, hidden_channels=16, num_classes=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    # In train_node_class.py, inside the main block:
    # We give higher weight to the smaller classes (like Executives and Accountants)
    weights = torch.tensor([2.0, 1.0, 2.0, 2.0, 5.0, 0.5, 1.0]) 
    criterion = torch.nn.CrossEntropyLoss(weight=weights)               

    def train():
        model.train()
        optimizer.zero_grad()
        
        # 1. Make a prediction for EVERYONE in the network
        out = model(data.x, data.edge_index)
        
        # 2. Calculate the loss ONLY on our 17 VIPs (using the train_mask)
        # The model ignores the 15,000 unknowns while it is learning
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()
        return loss.item()

    print("\n3. Training the Model on our VIPs...")
    for epoch in range(1, 501):
        loss = train()
        if epoch % 20 == 0:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
            
    print("\nTraining Complete! The GNN has learned what makes an Executive vs. a Trader.")


    # --- ADD THIS TO THE BOTTOM OF train_node_class.py ---

print("\n4. Predicting roles for the rest of the company...")
model.eval()

# @torch.no_grad() turns off the training math to save memory
with torch.no_grad():
    # Pass everyone through the model one last time
    out = model(data.x, data.edge_index)
    
    # The model outputs 4 scores (one for each job) per person. 
    # .argmax(dim=1) picks the job with the highest score!
    predictions = out.argmax(dim=1)

print("5. Rebuilding name dictionary to read the results...")
import pandas as pd
import email
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
unique_emails = set([src for src, dst in edges_text] + [dst for src, dst in edges_text])
node_mapping = {email_addr: i for i, email_addr in enumerate(unique_emails)}

# Reverse the mapping: Turn ID numbers back into Email Addresses
id_to_email = {v: k for k, v in node_mapping.items()}
role_names = {0: "Executive", 1: "Trader/Quant", 2: "Legal/Gov", 3: "Admin/Support", 4: "Accountant", 5: "External/Vendor", 6: "Automated/Bot"}

print("\n=======================================================")
print("  TOP 25 MOST ACTIVE UNLABELED EMPLOYEES & PREDICTIONS")
print("=======================================================")

# We want to print the most active people, not random 1-email accounts.
# data.x contains the log-normalized email counts. 
# summing them gives us an "Activity Score" for each person.
activity_levels = data.x.sum(dim=1)
top_active_nodes = activity_levels.argsort(descending=True)

count = 0
for node_id in top_active_nodes:
    node_id = node_id.item()
    
    # ONLY print people who were NOT in our original 17-person answer key
    if not data.train_mask[node_id]:
        email_addr = id_to_email[node_id]
        clean_name = email_addr.replace("@enron.com", "")
        predicted_role = role_names[predictions[node_id].item()]
        
        # Format the output to look like a clean table
        print(f"{clean_name:30} -> Predicted: {predicted_role}")
        
        count += 1
        if count >= 25:
            break