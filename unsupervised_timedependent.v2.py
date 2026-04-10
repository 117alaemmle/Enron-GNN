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

print("1. Parsing 2001-Q3 Emails...")
df = pd.read_csv('./Kaggle_DataSet/emails.csv', nrows=200000)

q_emails = []
for raw_message in df['message']:
    msg = email.message_from_string(raw_message)
    date_str = msg.get('Date')
    sender = msg.get('From')
    receivers_raw = msg.get('To')
    
    if date_str and sender and receivers_raw:
        try:
            dt = parser.parse(date_str)
            # Filter specifically for 2001-Q3 (July, August, September)
            if dt.year == 2001 and dt.month in [7, 8, 9]:
                sender = sender.strip().lower()
                receivers = [r.strip().lower() for r in receivers_raw.replace('\n', '').split(',')]
                
                body = msg.get_payload()
                if isinstance(body, list): body = body[0].get_payload()
                body = str(body).replace('\n', ' ')
                
                for r in receivers:
                    if r: q_emails.append({"src": sender, "dst": r, "text": body})
        except:
            continue

print(f"Found {len(q_emails)} emails for 2001-Q3.")

# Filter for active nodes (degree >= 3)
node_counts = defaultdict(int)
for e in q_emails:
    node_counts[e['src']] += 1
    node_counts[e['dst']] += 1
    
core_nodes = {node for node, count in node_counts.items() if count >= 6}
edges = [(e['src'], e['dst'], e['text']) for e in q_emails if e['src'] in core_nodes and e['dst'] in core_nodes]

unique_emails = sorted(list(core_nodes))
mapping = {em: i for i, em in enumerate(unique_emails)}
id_to_email = {i: em for i, em in enumerate(unique_emails)}

src = [mapping[s] for s, r, t in edges]
dst = [mapping[r] for s, r, t in edges]
edge_index = torch.tensor([src, dst], dtype=torch.long)

out_deg = degree(edge_index[0], num_nodes=len(unique_emails))
in_deg = degree(edge_index[1], num_nodes=len(unique_emails))
x = torch.log1p(torch.stack([out_deg, in_deg], dim=1))

print("2. Training Graph Autoencoder...")
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

model = GAE(GCNEncoder(x.shape[1], 8))
opt = torch.optim.Adam(model.parameters(), lr=0.01)

for _ in range(150):
    model.train()
    opt.zero_grad()
    z = model.encode(x.float(), edge_index)
    loss = model.recon_loss(z, edge_index)
    loss.backward()
    opt.step()

print("3. Clustering and Extracting Topics...")
model.eval()
with torch.no_grad():
    z = model.encode(x.float(), edge_index)
    z_np = z.cpu().numpy()

num_clusters = 10
clusters = KMeans(n_clusters=num_clusters, n_init=10, random_state=42).fit_predict(z_np)

cluster_texts = {i: [] for i in range(num_clusters)}
for s, r, text in edges:
    cluster_texts[clusters[mapping[s]]].append(text)

# TF-IDF Keyword Extraction
custom_stops = list(TfidfVectorizer(stop_words='english').get_stop_words()) + [
    'enron', 'com', 'subject', 'forwarded', 'pm', 'am', 'cc', 'http', 'www', 'ect', 'hou'
]
vectorizer = TfidfVectorizer(stop_words=custom_stops, max_features=1000)

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

print("4. Reducing Dimensions for Visualization...")
z_2d = TSNE(n_components=2, perplexity=30, max_iter=1000).fit_transform(z_np)

print("5. Generating Interactive Plot...")
viz_df = pd.DataFrame({
    'x': z_2d[:, 0],
    'y': z_2d[:, 1],
    'Email': [id_to_email[i] for i in range(len(unique_emails))],
    'Cluster ID': [f"Group {clusters[i]}" for i in range(len(unique_emails))],
    'Topics': [cluster_labels[clusters[i]] for i in range(len(unique_emails))],
    'Is Internal': ["Yes" if "@enron.com" in id_to_email[i] else "No" for i in range(len(unique_emails))]
})

# Make internal Enron employees larger to stand out against external noise
viz_df['Size'] = viz_df['Is Internal'].apply(lambda x: 12 if x == "Yes" else 4)

fig = px.scatter(
    viz_df, x='x', y='y', 
    color='Cluster ID',
    size='Size',
    hover_name='Email',
    hover_data={'Topics': True, 'x': False, 'y': False, 'Size': False, 'Is Internal': False},
    title="Enron Semantic Silos: 2001-Q3 (Pre-Bankruptcy)",
    template="plotly_dark"
)

# Improve legend and layout
fig.update_layout(legend_title_text='Communication Silos')
fig.show()