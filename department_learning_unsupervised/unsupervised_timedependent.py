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
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# Create output directory for our interactive maps
output_dir = "department_learning_unsupervised/interactive_department_maps"
os.makedirs(f"{output_dir}", exist_ok=True)

print("1. Parsing Emails by Quarter...")
df = pd.read_csv('./Kaggle_DataSet/emails.csv')

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
            
            body = msg.get_payload()
            if isinstance(body, list): body = body[0].get_payload()
            body = str(body).replace('\n', ' ')
            
            for r in receivers:
                if r: quarterly_data[q_key].append({"src": sender, "dst": r, "text": body})
        except:
            continue

# Sort the quarters chronologically
all_quarters = sorted(list(quarterly_data.keys()))
print(f"Found {len(all_quarters)} active quarters: {', '.join(all_quarters)}")

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

# 2. THE MASTER LOOP
for q_label in all_quarters:
    print(f"\n{'='*50}\n Processing: {q_label}\n{'='*50}")
    
    q_emails = quarterly_data[q_label]
    
    # FILTER: Must be involved in at least 5 emails this quarter
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
    mapping = {em: i for i, em in enumerate(unique_emails)}
    id_to_email = {i: em for i, em in enumerate(unique_emails)}
    
    src = [mapping[s] for s, r, t in edges]
    dst = [mapping[r] for s, r, t in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    out_deg = degree(edge_index[0], num_nodes=len(unique_emails))
    in_deg = degree(edge_index[1], num_nodes=len(unique_emails))
    x = torch.log1p(torch.stack([out_deg, in_deg], dim=1))
    
    # Train GAE
    model = GAE(GCNEncoder(x.shape[1], 8))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for _ in range(150):
        model.train()
        opt.zero_grad()
        z = model.encode(x.float(), edge_index)
        loss = model.recon_loss(z, edge_index)
        loss.backward()
        opt.step()

    # Cluster
    model.eval()
    with torch.no_grad():
        z = model.encode(x.float(), edge_index)
        z_np = z.cpu().numpy()

    num_clusters = min(10, len(unique_emails) // 5) # Ensure we don't have more clusters than people
    clusters = KMeans(n_clusters=num_clusters, n_init=10, random_state=42).fit_predict(z_np)

    cluster_texts = {i: [] for i in range(num_clusters)}
    for s, r, text in edges:
        cluster_texts[clusters[mapping[s]]].append(text)

    # NLP Extraction
    # TIP: You can add "noise" words like 'davis', 'california', 'gas' here to force the model to look deeper!
   # NLP Extraction
    custom_stops = list(TfidfVectorizer(stop_words='english').get_stop_words()) + [
        'enron', 'com', 'subject', 'forwarded', 'pm', 'am', 'cc', 'http', 'www', 'ect', 'hou', 'corp', 'mailto', 'thanks', 'mail', 'aol', 'yahoo', 'new', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
        'hello'
    ]
    
    # We add a token_pattern that forces words to be strictly alphabetical (no numbers!)
    vectorizer = TfidfVectorizer(
        stop_words=custom_stops, 
        max_features=1000,
        token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b' 
    )

    cluster_labels = {}
    for i in range(num_clusters):
        if not cluster_texts[i]:
            cluster_labels[i] = "No Data"
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
        title=f"Enron Semantic Silos: {q_label}", template="plotly_dark"
    )
    
    # Save to file instead of opening browser
    out_path = os.path.join(output_dir, f"enron_silos_{datetime.now().strftime('%Y%m%d-%H%M')}_{q_label}.html")
    fig.write_html(out_path)
    print(f"   -> Saved map to {out_path}")

print("\nDone! Check the 'interactive_department_maps' folder for your interactive maps.")