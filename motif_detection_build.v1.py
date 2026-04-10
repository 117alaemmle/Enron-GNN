#This code creates a PY data set which has an additional variable: the classification of individual executives based on their real-life roles. 
#We will use this data set to train a GNN to predict the roles of everyone in the graph. "Roles" are not things like 'trader' or 'lawyer' but rather broader categories like "Executive", "Administrator", "General Employee", "Manager", etc. These are types of employee which exist across departments.
#We will determine departments via unsupervised learning. For right now we determine the type of employee each person is, regardless of department.
#This uses motif detection.

import pandas as pd
import email
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

print("1. Loading data (Scanning more rows to catch our VIPs!)...")
# We bumped nrows to 50,000 to make sure we capture our executives' emails
df = pd.read_csv('./Kaggle_DataSet/emails.csv', nrows=50000)

print("2. Parsing emails and mapping to IDs...")
edges_text = []
for raw_message in df['message']:
    msg = email.message_from_string(raw_message)
    sender = msg.get('From')
    receivers_raw = msg.get('To')
    
    if sender and receivers_raw:
        sender = sender.strip().lower()
        receivers = [r.strip().lower() for r in receivers_raw.replace('\n', '').split(',')]
        for receiver in receivers:
            if receiver:
                edges_text.append((sender, receiver))

unique_emails = set([src for src, dst in edges_text] + [dst for src, dst in edges_text])
node_mapping = {email_addr: i for i, email_addr in enumerate(unique_emails)}
num_nodes = len(unique_emails)

source_nodes = [node_mapping[src] for src, dst in edges_text]
target_nodes = [node_mapping[dst] for src, dst in edges_text]
edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

print("3. Building Node Features (X)...")
out_degree = degree(edge_index[0], num_nodes=num_nodes)
in_degree = degree(edge_index[1], num_nodes=num_nodes)
# Applying the log normalization right from the start this time!
x = torch.log1p(torch.stack([out_degree, in_degree], dim=1))

print("4. Injecting the Enron VIP Labels...")
# Role Key:
# 0: Executive/C-Suite
# 1: Trading/Quantitative Management
# 2: Legal/Regulatory/Gov Affairs
# 3: Executive Assistants/Admin

enron_labels = {
    # ==========================================
    # MANUALLY CURATED LABELS
    # ==========================================
    # --- 0: EXECUTIVES ---
    "kenneth.lay": 0,      # CEO / Chairman
    "jeff.skilling": 0,    # CEO / President
    "greg.whalley": 0,     # President (post-Skilling)
    "rick.buy": 0,         # Chief Risk Officer
    "richard.causey": 0,   # Chief Accounting Officer
    "david.delainey": 0,   # CEO
    "rod.hayslett": 0,     # CFO of Enron Transportation Services (AL)
    
    # --- 1: TRADING & QUANT ---
    "vince.kaminski": 1,   # Head of Quantitative Research
    "louise.kitchen": 1,   # Head of EnronOnline
    "john.lavorato": 1,    # CEO of Enron America (Head Trader)
    "mike.grigsby": 1,     # VP of Trading
    
    # --- 2: LEGAL & REGULATORY ---
    "steven.kean": 2,      # Chief of Staff / Gov Affairs
    "richard.shapiro": 2,  # VP Regulatory Affairs
    "james.derrick": 2,    # General Counsel
    "mark.haedicke": 2,    # Managing Director, Legal
    "jeff.dasovich": 2,    # Gov Affairs (Manual override from text miner)
    
    # --- 3: THE ADMINS (The Secret Hubs) ---
    "rosalee.fleming": 3,  # Ken Lay's Assistant
    "sherri.sera": 3,      # Jeff Skilling's Assistant
    "shirley.crenshaw": 3, # Vince Kaminski's Assistant
    "liz.taylor": 3,       # Greg Whalley's Assistant

    # --- 4: ACCOUNTANTS ---
    "sherron.watkins": 4,

    # --- 5: EXTERNAL/VENDOR ---
    "mjmoreland@aep.com": 5,
    "jackie.scardello@compaq.com": 5,
    "mstewart3@officedepot.com": 5,
    "dm-dmcn-help@dmlogix.com": 5, 
    "dbeck2land@dellnet.com": 5,

    # --- 6: AUTOMATED/BOTS & BROADCASTS ---
    "helpdesk": 6,
    "technology.enron": 6,
    "public.relations": 6,          # Re-assigned from text miner
    "enron.announcements": 6,       # Re-assigned from text miner
    "announcements.enron": 6,       # Re-assigned from text miner
    "chairman.enron": 6,            # Re-assigned from text miner
    "office.chairman": 6,           # Re-assigned from text miner
    "no.address": 6,                # System artifact
    "ethink": 6,                    # Internal campaign bot
    "40enron": 6,

    # ==========================================
    # AUTO-MINED FROM SIGNATURES
    # ==========================================
    # --- 0: EXECUTIVES (Mined as: president, ceo, vp) ---
    "ann.schmidt": 0,
    "coo.jeff": 0,
    "ken.skilling": 0,
    "lloyd.will": 0,
    "kay.chapman": 0,
    "david.oxley": 0,
    "lorna.brennan": 0,
    "j..edison": 0,
    "felicia.doan": 0,
    "karen.denne": 0,
    "gavin.dillingham": 0,
    "rob.bradley": 0,
    "sharonda.stephens": 0,
    "miyung.buster": 0,
    "dan.dorland": 0,
    "frank.ermis": 0,

    # --- 1: TRADERS & MANAGERS (Mined as: manager, trader, analyst, director) ---
    "phillip.allen": 1,
    "lisa.jacobson": 1,
    "karen.buckley": 1,
    "k..allen": 1,
    "john.arnold": 1,
    "colleen.koenig": 1,
    "jeff.youngflesh": 1,
    "matt.harris": 1,
    "kim.godfrey": 1,
    "sarah-joy.hunter": 1,
    "david.forster": 1,
    "m..schmidt": 1,
    "ravi.thuraisingham": 1,
    "johnny.palmer": 1,
    "r..harrington": 1,
    "jeff.bartlett": 1,
    "robert.badeer": 1,
    "eric.bass": 1,
    "tom.wilbeck": 1,
    "stacey.dempsey": 1,
    "alexandra.saler": 1,
    "gwendolyn.gray": 1,
    "daniel.diamond": 1,
    "christopher.watts": 1,
    "kayne.coulter": 1,
    "daniel.muschar": 1,
    "sally.beck": 1,
    "sheila.glover": 1,
    "mary.solmonson": 1,
    "lexi.elliott": 1,
    "celeste.roberts": 1,
    "ted.murphy": 1,
    "shona.wilson": 1,
    "hector.mcloughlin": 1,
    "brent.price": 1,
    "cheryl.kuehl": 1,
    "debbie.flores": 1,
    "sheri.thomas": 1,
    "barry.pearce": 1,
    "george.hope": 1,
    "annemarie.allex": 1,
    "mechelle.atwood": 1,
    "robert.benson": 1,
    "chris.dorland": 1,
    "jeffrey.keeler": 1,
    "lynn.blair": 1,
    "shelley.corman": 1,
    "airam.arteaga": 1,
    "victor.lamadrid": 1,
    "charla.reese": 1,
    "wendy.conwell": 1,
    "kortney.brown": 1,
    "larry.campbell": 1,
    "george.robinson": 1,
    "marc.phillips": 1,
    "timothy.callahan": 1,
    "monika.causholli": 1,
    "catherine.mckalip-thompson": 1,
    "carmen.perez": 1,
    "susan.landwehr": 1,
    "sue.nord": 1,
    "lynnette.barnes": 1,
    "scott.bolton": 1,
    "dirk.vanulden": 1,
    "jeffery.fawcett": 1,
    "margo.reyna": 1,
    "stacey.bolton": 1,
    "rob.cone": 1,
    "l..petrochko": 1,
    "mike.curry": 1,
    "kate.cole": 1,
    "scott.dozier": 1,
    "tom.donohoe": 1,
    "daren.farmer": 1,

    # --- 2: LEGAL & GOV (Mined as: attorney, counsel) ---
    "jr..legal": 2,
    "tamara.black": 2,
    "janette.elbertson": 2,
    "vicki.sharp": 2,
    "michael.tribolet": 2,
    "michelle.cash": 2,
    "mark.greenberg": 2,
    "rob.walls": 2,
    "sean.crandall": 2,
    "james.steffes": 2,
    "jennifer.rudolph": 2,
    "susan.mara": 2,
    "mary.hain": 2,
    "stephanie.harris": 2,
    "stacy.dickson": 2,

    # --- 3: ADMIN/SUPPORT (Mined as: assistant, admin) ---
    "jae.black": 3,
    "rebecca.torres": 3,
    "kimberly.brown": 3,
    "patti.thompson": 3,
    "greg.piper": 3,
    "tina.spiller": 3,
    "paula.rieker": 3,
    "katherine.brown": 3,
    "laura.valencia": 3,
    "d..hogan": 3,
    "ginger.dernehl": 3,
    "alan.comnes": 3,
    "sarah.novosel": 3,
    "joseph.alamo": 3,
    "dana.davis": 3,
    "nicole.mendez": 3,
    "larry.pardue": 3
}

# --- REPLACING THE OLD MAPPING LOOP WITH STRICT DOMAIN FILTERING ---

# Create empty lists to hold our labels and training flags
node_y = []
train_mask = []

matched_labels = 0
external_count = 0

for email_addr in unique_emails:
    label = -1      # -1 means "Internal but role unknown"
    is_trained = False
    
    # 1. Check if the address is in our VIP/Mined dictionary
    # We check the prefix (jeff.skilling) and the full address
    prefix = email_addr.split('@')[0]
    
    if prefix in enron_labels:
        label = enron_labels[prefix]
        is_trained = True
        matched_labels += 1
    elif email_addr in enron_labels:
        label = enron_labels[email_addr]
        is_trained = True
        matched_labels += 1
        
    # 2. STRICT DOMAIN CHECK: If it's not Enron, it's definitely External (Category 5)
    # We mark is_trained = True so the GNN learns this as a "known" fact
    elif "@enron.com" not in email_addr:
        label = 5
        is_trained = True
        external_count += 1
        
    # Default 'unknown' internal nodes to 0 (the mask will prevent them from being used for training)
    node_y.append(label if label != -1 else 0)
    train_mask.append(is_trained)

# Convert our lists into PyTorch tensors
y = torch.tensor(node_y, dtype=torch.long)
train_mask = torch.tensor(train_mask, dtype=torch.bool)

print(f"6. Labeling Summary:")
print(f"   - Manually/Mined Labels: {matched_labels}")
print(f"   - Auto-Detected External: {external_count}")
print(f"   - Total Training Nodes: {matched_labels + external_count} out of {num_nodes}")

# --- Proceed to Saving ---
data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)
torch.save(data, 'enron_node_class_data.pt')
print("\nSuccess! Data saved to 'enron_node_class_data.pt'")