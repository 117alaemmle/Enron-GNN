import pandas as pd
import email
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree
import os
from dateutil import parser

# Create a folder for our time slices
if not os.path.exists('time_slices'):
    os.makedirs('time_slices')

print("1. Loading full dataset for time-slicing...")
# We scan 100,000 rows to get a good temporal spread (late 1999 - late 2001)
df = pd.read_csv('./Kaggle_DataSet/emails.csv', nrows=100000)

print("2. Parsing dates and grouping emails...")
# We want to group emails by Year-Month (e.g., '2000-05')
monthly_edges = {}

for raw_message in df['message']:
    msg = email.message_from_string(raw_message)
    # Extract Date
    date_str = msg.get('Date')
    sender = msg.get('From')
    receivers_raw = msg.get('To')
    
    if date_str and sender and receivers_raw:
        try:
            # Standardize the date
            dt = parser.parse(date_str)
            if dt.year < 1998 or dt.year > 2002: continue # Filter outliers
            month_key = f"{dt.year}-{dt.month:02d}"
            
            sender = sender.strip().lower()
            receivers = [r.strip().lower() for r in receivers_raw.replace('\n', '').split(',')]
            
            if month_key not in monthly_edges:
                monthly_edges[month_key] = []
                
            for r in receivers:
                if r: monthly_edges[month_key].append((sender, r.strip().lower()))
        except:
            continue

print(f"3. Building {len(monthly_edges)} monthly graphs...")

for month, edges in monthly_edges.items():
    if len(edges) < 100: continue # Skip months with too little data
    
    # Standard GNN Graph Building
    nodes = set([s for s, r in edges] + [r for s, r in edges])
    mapping = {node: i for i, node in enumerate(nodes)}
    
    src = [mapping[s] for s, r in edges]
    dst = [mapping[r] for s, r in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # Simple features: In/Out Degree for that month
    out_deg = degree(edge_index[0], num_nodes=len(nodes))
    in_deg = degree(edge_index[1], num_nodes=len(nodes))
    x = torch.log1p(torch.stack([out_deg, in_deg], dim=1))
    
    data = Data(x=x, edge_index=edge_index)
    torch.save(data, f'time_slices/graph_{month}.pt')

print("\nDone! Check your 'time_slices' folder. You now have a month-by-month history of Enron.")