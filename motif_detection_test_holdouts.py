import torch
import pandas as pd
import email
from motif_detection_train import MotifGNN

print("1. Re-building email mapping...")
df = pd.read_csv('./Kaggle_DataSet/emails.csv', nrows=500000)
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
node_mapping = {email_addr: i for i, email_addr in enumerate(unique_emails)}

print("2. Loading the graph and trained model...")
data = torch.load('motif_enron_data.pt', weights_only=False)
model = MotifGNN(data.x.shape[1], 32, 6)
model.load_state_dict(torch.load('motif_model_weights.pth'))
model.eval()

role_names = {
    0: "Executive/Board", 1: "Manager/Director", 2: "Specialist/Trader",
    3: "General Staff/Analyst", 4: "Admin/Support", 5: "External"
}

# The people we hid from the training data
holdouts = {
    #Executives:
    "greg.whalley": "Executive/Board", 
    "jeff.skilling": "Executive/Board", 
    "richard.buy": "Executive/Board",
    #Managers
    "tim.belden": "Manager/Director",
    "joe.deffner": "Manager/Director",
    "michael.kopper": "Manager/Director",
    #Specialists
    "john.arnold": "Specialist/Trader",
    "jordan.mintz": "Specialist/Trader",
    "sharon.butcher": "Specialist/Trader",
    #General Staff
    "wanda.curry": "General Staff/Analyst",
    "kent.castleman": "General Staff/Analyst",
    "mary.clark": "General Staff/Analyst",
    #Admins
    "rosalee.fleming": "Admin/Support"
}

print("3. Generating Predictions for Holdouts...")
with torch.no_grad():
    logits = model(data.x.float(), data.edge_index)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

print(f"\n{'EMAIL PREFIX': <20} | {'ACTUAL ROLE': <20} | {'PREDICTED ROLE': <25} | {'CONF'}")
print("-" * 80)

for prefix, actual_role in holdouts.items():
    # Find all email addresses that start with this prefix
    for email_addr, node_id in node_mapping.items():
        if email_addr.startswith(prefix + "@"):
            pred_role = preds[node_id].item()
            confidence = probs[node_id][pred_role].item()
            pred_str = role_names.get(pred_role, "Unknown")
            
            print(f"{prefix: <20} | {actual_role: <20} | {pred_str: <25} | {confidence:.2f}")