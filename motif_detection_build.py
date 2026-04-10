#Run FIRST to build the motif dataset.
import pandas as pd
import email
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

print("1. Loading Enron Corpus (Building Hierarchical Graph)...")
df = pd.read_csv('./Kaggle_DataSet/emails.csv', nrows=500000)

# 2. EMAIL PARSING & MAPPING
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
num_nodes = len(unique_emails)

source_nodes = [node_mapping[src] for src, dst in edges_text]
target_nodes = [node_mapping[dst] for src, dst in edges_text]
edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

# ==========================================
# ROLE-BASED LABELS FOR MOTIF DETECTION
# ==========================================
# 0: Executives & Board (The Hubs - High In-Degree, broadcast to VPs)
# 1: Managers & Directors (The Bridges - Connect the floor to the C-Suite)
# 2: Specialists & Traders (The Cliques - High density internal chatter)
# 3: General Staff & Analysts (The Leaves - Lower volume, report to managers)
# 4: Admins & Support (The Stars - Massive Out-Degree broadcasts, low replies)

role_labels = {
    # --- 0: EXECUTIVES & BOARD MEMBERS (The Hubs) --- Removed #"greg.whalley": 0, "jeff.skilling": 0, "richard.buy": 0,
    "kenneth.lay": 0,   
    "mark.frevert": 0, "cliff.baxter": 0, "mick.seidl": 0, "andrew.fastow": 0, 
    "richard.causey": 0, "john.lavorato": 0, "louise.kitchen": 0,
    "kenneth.rice": 0, "kevin.hannon": 0, "joe.hirko": 0, "lou.pai": 0, 
    "dave.delainey": 0, "rebecca.mark": 0, "joe.sutton": 0, "rebecca.mcdonald": 0, 
    "john.sherriff": 0, "stanley.horton": 0, 
    "jeffrey.sherrick": 0, "forresthoglund": 0, "william.powers": 0,
    "wgramm": 0, "john.urquhart": 0,

    # --- 1: MANAGERS, DIRECTORS & VPs (The Bridges) --- Removed "timothy.belden": 1, "joe.deffner": 1, "michael.kopper": 1,
    "jeffrey.mcmahon": 1, "ben.glisan": 1, "raymond.bowen": 1,  
    "bill.gathmann": 1, "lfastow": 1, "bill.brown": 1, "james.timmins": 1, 
    "kathy.lynn": 1, "shirley.hudler": 1, "tim.despain": 1, 
    "mike.jakubik": 1, "kelly.boots": 1, "woytek.david": 1, "wes.colwell": 1, 
    "ryan.siurek": 1, "rodney.faldyn": 1, "bob.butts": 1,  
    "mike.mcconnell": 1, "jeffrey.shankman": 1, "john.forney": 1, "don.black": 1, 
    "david.cox": 1, "jim.fallon": 1, "kevin.howard": 1, 
    "michael.krautz": 1, "larry.lawyer": 1, "tom.white": 1, "jeremy.blachman": 1, 
    "margaret.ceconi": 1, "jimmie.williams": 1, "javier.li": 1, "james.c.alexander": 1, "bob.kelly": 1, "michael.brown": 1, "amanda.martin": 1, 
    "frank.stabler": 1, "jere.overdyke": 1, "colin.skellett": 1, "mark.palmer": 1, 
    "mark.koenig": 1, "kean.steve": 1, "cindy.olson": 1, 
    "rebecca.carter": 1, "beth.tilney": 1, "mary.joyce": 1,
    "david.oxley": 1, "robert.jones": 1, "danny.mccarty": 1, "john.esslinger": 1, 
    "john.harding": 1, "vince.kaminski": 1, "sherron.watkins": 1, "greg.piper": 1, 
    "allan.sommer": 1, "mark.muller": 1, "gene.humphrey": 1,
    "alan.quaintance": 1, "clint.walden": 1,

    # --- 2: SPECIALISTS, TRADERS & LEGAL (The Cliques) --- "john.arnold": 2, "jordan.mintz": 2, "sharon.butcher": 2,
    "angela.schwarz": 2, "james.derrick": 2, 
    "mark.haedicke": 2, "kristina.mordaunt": 2, "richard.sanders": 2, 
    "rob.walls": 2, "rex.rogers": 2, "stuart.zisman": 2, "christian.yoder": 2, 
    "carol.st.clair": 2, "scott.sefton": 2,

    # --- 3: GENERAL STAFF, ANALYSTS & ACCOUNTANTS (The Leaves) --- Removed: "wanda.curry": 3, "kent.castleman": 3, "mary.clark": 3,
    "kevin.jordan": 3, "charles.weiss": 3, "trushar.patel": 3, "stinson.gibner": 3, 
    "vasant.shanbhogue": 3, "rakesh.bharati": 3, "kevin.kindall": 3, "mark.lay": 3, 
    "keith.power": 3, "steve.pearlman": 3, "joannie.williamson": 3, 
    "ding.yuan": 3, "s..galvan": 3, "ron.baker": 3,
    "george.mckean": 3, "sheila.knudsen": 3, 
    "cheryl.lipshutz": 3,

    # --- 4: ADMINS & SUPPORT (The Broadcast Stars) --- "rosalee.fleming": 4,
    "sherri.sera": 4, "kim.garcia": 4
}

# 4. NODE LABELING & MASKING
y = torch.full((num_nodes,), -1, dtype=torch.long)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)

matched = 0
external = 0

for email_addr, node_id in node_mapping.items():
    prefix = email_addr.split('@')[0]
    
    # Check if they are in our known role list
    if prefix in role_labels:
        y[node_id] = role_labels[prefix]
        train_mask[node_id] = True
        matched += 1
    # Auto-label external nodes as Category 5 (The Outside World)
    elif "@enron.com" not in email_addr:
        y[node_id] = 5
        #train_mask[node_id] = True #Due to a profusion of external nodes causing the model to get lazy and just predict "External" for everyone, we will NOT include these in the training set.
        external += 1

# Default remaining unknowns to 3 (General Staff) but KEEP mask False
y[y == -1] = 3 

print(f"Graph Statistics:")
print(f" - Total Nodes: {num_nodes}")
print(f" - Known Internal Roles: {matched}")
print(f" - Known External Nodes: {external}")

# 5. STRUCTURAL FEATURES (X)
# Motif detection relies heavily on degree distribution
out_deg = degree(edge_index[0], num_nodes=num_nodes)
in_deg = degree(edge_index[1], num_nodes=num_nodes)
x = torch.log1p(torch.stack([out_deg, in_deg], dim=1))

# 6. PACKAGE DATA
data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)
torch.save(data, 'motif_enron_data.pt')
print("\nSuccess! Saved to 'motif_enron_data.pt'.")