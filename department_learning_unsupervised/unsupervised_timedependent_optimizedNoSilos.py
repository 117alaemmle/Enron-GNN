from datetime import datetime
import os
import torch
import pandas as pd
import email
from dateutil import parser
from collections import defaultdict
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.utils import degree
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import plotly.express as px
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Create output directory for our interactive maps
output_dir = "department_learning_unsupervised/interactive_department_maps"
os.makedirs(output_dir, exist_ok=True)

print("1. Parsing Emails by Quarter...")
# Bulletproof path loading
SCRIPT_DIR = Path(__file__).parent
file_path = SCRIPT_DIR.parent / 'Kaggle_DataSet' / 'emails.csv'

# Fallback if the above doesn't match your exact folder structure
if not file_path.exists():
    file_path = SCRIPT_DIR / 'Kaggle_DataSet' / 'emails.csv'
    if not file_path.exists():
        file_path = '../Kaggle_DataSet/emails.csv'

df = pd.read_csv(file_path)

quarterly_data = defaultdict(list)

for raw_message in df['message']:
    msg = email.message_from_string(raw_message)
    date_str = msg.get('Date')
    sender = msg.get('From')
    receivers_raw = msg.get('To')
    
    if date_str and sender and receivers_raw:
        try:
            dt = parser.parse(date_str)
            if dt.year < 1998 or dt.year > 2002: continue
            
            quarter = (dt.month - 1) // 3 + 1
            q_key = f"{dt.year}-Q{quarter}"
            
            sender = sender.strip().lower()
            receivers = [r.strip().lower() for r in receivers_raw.replace('\n', '').split(',')]
            
            # Extract email body text
            body = msg.get_payload()
            if isinstance(body, list): body = body[0].get_payload()
            body = str(body).replace('\n', ' ')
            
            for r in receivers:
                if r: quarterly_data[q_key].append({"src": sender, "dst": r, "text": body})
        except:
            continue

all_quarters = sorted(list(quarterly_data.keys()))
print(f"Found {len(all_quarters)} active quarters to analyze.\n")

# 2. DEFINE THE ENCODER
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

# 3. THE EVOLUTION & CLUSTERING LOOP
for q_label in all_quarters:
    print(f"\n{'='*60}")
    print(f" TIME SLICE: {q_label}")
    print(f"{'='*60}")
    
    q_emails = quarterly_data[q_label]
    if not q_emails: continue
        
    # Filter for active nodes (degree >= 5) to keep the clusters tight
    node_counts = defaultdict(int)
    for e in q_emails:
        node_counts[e['src']] += 1
        node_counts[e['dst']] += 1
        
    core_nodes = {node for node, count in node_counts.items() if count >= 5}
    edges = [(e['src'], e['dst'], e['text']) for e in q_emails if e['src'] in core_nodes and e['dst'] in core_nodes]
    
    if len(core_nodes) < 20:
        print("Not enough core activity this quarter. Skipping...")
        continue
    
    unique_emails = sorted(list(core_nodes))
    id_to_email = {i: em for i, em in enumerate(unique_emails)}
    mapping = {em: i for i, em in enumerate(unique_emails)}
    
    src = [mapping[s] for s, r, t in edges]
    dst = [mapping[r] for s, r, t in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    out_deg = degree(edge_index[0], num_nodes=len(unique_emails))
    in_deg = degree(edge_index[1], num_nodes=len(unique_emails))
    x = torch.log1p(torch.stack([out_deg, in_deg], dim=1))
    
    # Train GAE
    model = GAE(GCNEncoder(x.shape[1], 8))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x.float(), edge_index)
        loss = model.recon_loss(z, edge_index)
        loss.backward()
        optimizer.step()
        
    # Extract Coordinates
    model.eval()
    with torch.no_grad():
        z = model.encode(x.float(), edge_index)
        z_np = z.cpu().numpy()
        
    # --- THE SILHOUETTE OPTIMIZER ---
    print("   -> Optimizing Silo Count (Testing k=3 to 15)...")
    max_k = min(20, len(unique_emails) // 10)
    best_k = 3
    best_score = -1.0
    best_kmeans = None

    if max_k >= 3:
        for k in range(3, max_k + 1):
            temp_kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            temp_labels = temp_kmeans.fit_predict(z_np)
            
            # Silhouette requires at least 2 clusters
            if 1 < len(set(temp_labels)) < len(z_np):
                score = silhouette_score(z_np, temp_labels)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_kmeans = temp_kmeans

    # Fallback just in case optimization fails
    if best_kmeans is None:
        best_k = max(2, min(5, max_k))
        best_kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(z_np)

    clusters = best_kmeans.labels_
    print(f"   ✅ Optimal Silos Found: {best_k} (Silhouette Clarity Score: {best_score:.3f})")
    
    # Group text by cluster
    cluster_texts = {i: [] for i in range(best_k)}
    for s, r, text in edges:
        cluster_texts[clusters[mapping[s]]].append(text)

    # 4. NLP TOPIC EXTRACTION
    custom_stops = list(TfidfVectorizer(stop_words='english').get_stop_words()) + [
        'enron', 'com', 'subject', 'forwarded', 'pm', 'am', 'cc', 'http', 'www', 'ect', 'hou', 'corp', 'mailto', 'thanks', 'mail', 'aol', 'yahoo', 'new', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
        'hello', 'attached'
    ]
    
    # Restrict to alphabetical strings to filter out years, dollars, phone numbers
    vectorizer = TfidfVectorizer(
        stop_words=custom_stops, 
        max_features=1000,
        token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b' 
    )
    
    cluster_labels = {}
    for i in range(best_k):
        if not cluster_texts[i]:
            cluster_labels[i] = "General Chatter"
            continue
        try:
            tfidf_matrix = vectorizer.fit_transform([" ".join(cluster_texts[i])])
            feature_names = vectorizer.get_feature_names_out()
            dense = tfidf_matrix.todense()
            top_words = [feature_names[j] for j in dense[0].argsort().tolist()[0][-5:]][::-1]
            cluster_labels[i] = ", ".join(top_words).title()
        except ValueError:
            cluster_labels[i] = "General Chatter"

    # Reduce and Plot
    z_2d = TSNE(n_components=2, perplexity=min(30, len(z_np)-1), max_iter=1000).fit_transform(z_np)
    
    viz_df = pd.DataFrame({
        'x': z_2d[:, 0], 'y': z_2d[:, 1],
        'Email': [id_to_email[i] for i in range(len(unique_emails))],
        'Topics': [cluster_labels[clusters[i]] for i in range(len(unique_emails))],
        'Is Internal': ["Yes" if "@enron.com" in id_to_email[i] else "No" for i in range(len(unique_emails))]
    })
    
    viz_df['Size'] = viz_df['Is Internal'].apply(lambda x: 12 if x == "Yes" else 4)

    fig = px.scatter(
        viz_df, x='x', y='y', color='Topics', size='Size', hover_name='Email',
        hover_data={'Topics': True, 'x': False, 'y': False, 'Size': False, 'Is Internal': False},
        title=f"Enron Semantic Silos: {q_label} (Optimal Silos: {best_k})", template="plotly_dark"
    )
    
    # Save to file
    timestamp_str = datetime.now().strftime('%Y%m%d-%H%M')
    out_path = os.path.join(output_dir, f"enron_silos_{timestamp_str}_{q_label}.html")
    fig.write_html(out_path)
    print(f"   -> Saved optimized map to {out_path}")

print("\nDone! Check the 'interactive_department_maps' folder for your interactive maps.")