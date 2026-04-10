import os
import pandas as pd
import email
from dateutil import parser
from collections import defaultdict
import networkx as nx
import numpy as np
import torch
from torch_geometric.nn import GCNConv, GAE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# --- SET UP HTML OUTPUT ---
output_dir = "MotifsUnsupervisedOutput"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M")
html_filename = os.path.join(output_dir, f"motifs_unsupervised_{timestamp}.html")

html_content = """
<html>
<head>
    <title>Enron Temporal Archetypes</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1e1e1e; color: #d4d4d4; padding: 20px; }
        h1 { color: #569cd6; border-bottom: 2px solid #333; padding-bottom: 10px; }
        h2 { color: #4ec9b0; margin-top: 40px; }
        .cluster-box { background-color: #252526; border: 1px solid #444; border-radius: 5px; padding: 15px; margin-bottom: 20px; }
        .cluster-header { font-size: 1.2em; font-weight: bold; color: #c586c0; margin-bottom: 10px; }
        .vip-list { color: #dcdcaa; font-weight: bold; margin-bottom: 10px; }
        .archetype-list { color: #ce9178; font-family: 'Courier New', Courier, monospace; }
        ul { margin-top: 5px; }
        li { margin-bottom: 3px; }
    </style>
</head>
<body>
    <h1>🕵️‍♂️ Enron Corporate Archetypes: Quarterly Evolution</h1>
"""

print("1. Parsing Emails by Quarter (Loading Full Corpus)...")
# Removed nrows restriction to load the entire dataset
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
            if dt.year < 1999 or dt.year > 2002: continue 
            
            quarter = (dt.month - 1) // 3 + 1
            q_key = f"{dt.year}-Q{quarter}"
            
            sender = sender.strip().lower()
            receivers = [r.strip().lower() for r in receivers_raw.replace('\n', '').split(',')]
            
            for r in receivers:
                if r: quarterly_data[q_key].append((sender, r))
        except:
            continue

all_quarters = sorted(list(quarterly_data.keys()))
print(f"Found {len(all_quarters)} active quarters to analyze.\n")

# --- EXPANDED VIP DICTIONARY ---
known_vips = {
    # Executives
    "kenneth.lay": "Executive", "mark.frevert": "Executive", "cliff.baxter": "Executive", 
    "mick.seidl": "Executive", "andrew.fastow": "Executive", "richard.causey": "Executive", 
    "john.lavorato": "Executive", "louise.kitchen": "Executive", "kenneth.rice": "Executive", 
    "kevin.hannon": "Executive", "joe.hirko": "Executive", "lou.pai": "Executive", 
    "dave.delainey": "Executive", "rebecca.mark": "Executive", "joe.sutton": "Executive", 
    "rebecca.mcdonald": "Executive", "john.sherriff": "Executive", "stanley.horton": "Executive", 
    "jeffrey.sherrick": "Executive", "forresthoglund": "Executive", "william.powers": "Executive", 
    "wgramm": "Executive", "john.urquhart": "Executive",

    # Managers
    "jeffrey.mcmahon": "Manager", "ben.glisan": "Manager", "raymond.bowen": "Manager", 
    "bill.gathmann": "Manager", "lfastow": "Manager", "bill.brown": "Manager", 
    "james.timmins": "Manager", "kathy.lynn": "Manager", "shirley.hudler": "Manager", 
    "tim.despain": "Manager", "mike.jakubik": "Manager", "kelly.boots": "Manager", 
    "woytek.david": "Manager", "wes.colwell": "Manager", "ryan.siurek": "Manager", 
    "rodney.faldyn": "Manager", "bob.butts": "Manager", "mike.mcconnell": "Manager", 
    "jeffrey.shankman": "Manager", "john.forney": "Manager", "don.black": "Manager", 
    "david.cox": "Manager", "jim.fallon": "Manager", "kevin.howard": "Manager", 
    "michael.krautz": "Manager", "larry.lawyer": "Manager", "tom.white": "Manager", 
    "jeremy.blachman": "Manager", "margaret.ceconi": "Manager", "jimmie.williams": "Manager", 
    "javier.li": "Manager", "james.c.alexander": "Manager", "bob.kelly": "Manager", 
    "michael.brown": "Manager", "amanda.martin": "Manager", "frank.stabler": "Manager", 
    "jere.overdyke": "Manager", "colin.skellett": "Manager", "mark.palmer": "Manager", 
    "mark.koenig": "Manager", "kean.steve": "Manager", "cindy.olson": "Manager", 
    "rebecca.carter": "Manager", "beth.tilney": "Manager", "mary.joyce": "Manager", 
    "david.oxley": "Manager", "robert.jones": "Manager", "danny.mccarty": "Manager", 
    "john.esslinger": "Manager", "john.harding": "Manager", "vince.kaminski": "Manager", 
    "sherron.watkins": "Manager", "greg.piper": "Manager", "allan.sommer": "Manager", 
    "mark.muller": "Manager", "gene.humphrey": "Manager", "alan.quaintance": "Manager", 
    "clint.walden": "Manager",

    # Traders & Legal
    "angela.schwarz": "Trader/Legal", "james.derrick": "Trader/Legal", "mark.haedicke": "Trader/Legal", 
    "kristina.mordaunt": "Trader/Legal", "richard.sanders": "Trader/Legal", "rob.walls": "Trader/Legal", 
    "rex.rogers": "Trader/Legal", "stuart.zisman": "Trader/Legal", "christian.yoder": "Trader/Legal", 
    "carol.st.clair": "Trader/Legal", "scott.sefton": "Trader/Legal",

    # Staff
    "kevin.jordan": "Staff", "charles.weiss": "Staff", "trushar.patel": "Staff", 
    "stinson.gibner": "Staff", "vasant.shanbhogue": "Staff", "rakesh.bharati": "Staff", 
    "kevin.kindall": "Staff", "mark.lay": "Staff", "keith.power": "Staff", 
    "steve.pearlman": "Staff", "joannie.williamson": "Staff", "ding.yuan": "Staff", 
    "s..galvan": "Staff", "ron.baker": "Staff", "george.mckean": "Staff", 
    "sheila.knudsen": "Staff", "cheryl.lipshutz": "Staff",

    # Admins
    "sherri.sera": "Admin", "kim.garcia": "Admin"
}

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, 8)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

# 2. THE QUARTERLY EVOLUTION LOOP
for q_label in all_quarters:
    print(f"\n{'='*70}\n 📅 TIME SLICE: {q_label}\n{'='*70}")
    html_content += f"<h2>📅 Time Slice: {q_label}</h2>\n"
    
    q_edges = quarterly_data[q_label]
    
    G_raw = nx.DiGraph()
    G_raw.add_edges_from(q_edges)
    G_raw.remove_edges_from(nx.selfloop_edges(G_raw))
    
    core_nodes = [n for n, d in G_raw.degree() if d >= 5]
    G = G_raw.subgraph(core_nodes).copy()
    
    if len(G.nodes()) < 50:
        skip_msg = "Not enough core activity this quarter. Skipping..."
        print(skip_msg)
        html_content += f"<p><i>{skip_msg}</i></p>\n"
        continue
        
    unique_emails = list(G.nodes())
    node_mapping = {email_addr: i for i, email_addr in enumerate(unique_emails)}
    id_to_email = {i: em for em, i in node_mapping.items()}
    
    # Topological Math
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    pagerank = nx.pagerank(G, alpha=0.85)
    clustering = nx.clustering(G.to_undirected())
    
    try:
        hubs, authorities = nx.hits(G, max_iter=100, normalized=True)
    except nx.PowerIterationFailedConvergence:
        hubs = {node: 0.0 for node in G.nodes()}
        authorities = {node: 0.0 for node in G.nodes()}

    # Scale Features
    feature_matrix = []
    for node in unique_emails:
        feature_matrix.append([
            np.log1p(in_degrees[node]), np.log1p(out_degrees[node]),
            pagerank[node], clustering[node], hubs[node], authorities[node]
        ])
    
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(feature_matrix)
    x = torch.tensor(scaled_matrix, dtype=torch.float)
    
    src = [node_mapping[s] for s, dst in G.edges()]
    dst = [node_mapping[dst] for s, dst in G.edges()]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
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

    # K-Means Clustering
    model.eval()
    with torch.no_grad():
        z_np = model.encode(x.float(), edge_index).cpu().numpy()

    num_roles = min(6, len(unique_emails) // 10)
    kmeans = KMeans(n_clusters=num_roles, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(z_np)
    centers = kmeans.cluster_centers_
    
    # Evaluate and Print
    for i in range(num_roles):
        cluster_indices = np.where(clusters == i)[0]
        if len(cluster_indices) == 0: continue
        
        cluster_points = z_np[cluster_indices]
        centroid = centers[i].reshape(1, -1)
        distances = pairwise_distances(cluster_points, centroid).flatten()
        closest_idx = np.argsort(distances)
        
        print(f"\n🟢 CLUSTER {i} (Nodes: {len(cluster_indices)})")
        html_content += f'<div class="cluster-box">\n<div class="cluster-header">🟢 Cluster {i} (Nodes: {len(cluster_indices)})</div>\n'
        
        # Ground Truths
        vips_found = []
        for idx in range(len(cluster_indices)):
            email_addr = id_to_email[cluster_indices[idx]]
            prefix = email_addr.split('@')[0]
            if prefix in known_vips:
                vips_found.append(f"{prefix} ({known_vips[prefix]})")
                
        if vips_found:
            vip_str = ', '.join(vips_found)
            print(f"   ⭐ VIPs Found: {vip_str}")
            html_content += f'<div class="vip-list">⭐ VIPs Found: {vip_str}</div>\n'
        else:
            print("   ⭐ VIPs Found: (None)")
            html_content += '<div class="vip-list">⭐ VIPs Found: (None)</div>\n'
            
        # Top 5 Archetypes
        print(f"   📐 Top Archetypes:")
        html_content += '<div class="archetype-list">📐 Top Archetypes:<ul>\n'
        count = 0
        for idx in closest_idx:
            email_addr = id_to_email[cluster_indices[idx]]
            if "@enron.com" in email_addr:
                print(f"      - {email_addr: <28} (Dist: {distances[idx]:.4f})")
                html_content += f'<li>{email_addr} (Dist: {distances[idx]:.4f})</li>\n'
                count += 1
            if count >= 5: break
        
        html_content += '</ul></div>\n</div>\n'

# --- SAVE FINAL HTML REPORT ---
html_content += "</body></html>"
with open(html_filename, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"\n{'='*70}\n✅ Complete! Full results saved to:\n{html_filename}\n{'='*70}")