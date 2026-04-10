import torch
import pandas as pd
import email
from motif_detection_train import MotifGNN # Import your model class

print("1. Re-building email mapping (Scanning CSV to get names)...")
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

# Recreate the exact same sorted list we used in the build script
unique_emails = sorted(list(set([src for src, dst in edges_text] + [dst for src, dst in edges_text])))
id_to_email = {i: email_addr for i, email_addr in enumerate(unique_emails)}

print("2. Loading the graph data and the trained brain...")
data = torch.load('motif_enron_data.pt', weights_only=False)

# Re-initialize the model
model = MotifGNN(data.x.shape[1], 32, 6)
model.load_state_dict(torch.load('motif_model_weights.pth'))
model.eval()

# Human-readable dictionary
role_names = {
    0: "Executive/Board",
    1: "Manager/Director",
    2: "Specialist/Trader",
    3: "General Staff/Analyst",
    4: "Admin/Support",
    5: "External"
}

print("3. Generating Predictions...")
with torch.no_grad():
    logits = model(data.x.float(), data.edge_index)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

unlabeled_mask = ~data.train_mask

print(f"\n--- Checking Unlabeled Internal Enron Employees ---")
count = 0

for i in range(len(unlabeled_mask)):
    if unlabeled_mask[i]:
        email_addr = id_to_email[i]
        
        # We only want to look at how it classifies internal people we didn't train it on
        if "@enron.com" in email_addr:
            pred_role = preds[i].item()
            confidence = probs[i][pred_role].item()
            role_str = role_names.get(pred_role, "Unknown")
            
            # Formatted print statement for clean columns
            print(f"Email: {email_addr: <35} | Predicted: {role_str: <25} | Confidence: {confidence:.2f}")
            
            count += 1
            if count >= 25:  # Show the first 25 unknown internal employees
                break