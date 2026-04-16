import pandas as pd
import email
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

print("1. Loading dataset...")
SCRIPT_DIR = Path(__file__).parent
file_path = SCRIPT_DIR.parent / 'Kaggle_DataSet' / 'emails.csv'

# Fallback path
if not file_path.exists():
    file_path = SCRIPT_DIR / 'Kaggle_DataSet' / 'emails.csv'
    if not file_path.exists():
        file_path = '../Kaggle_DataSet/emails.csv'

df = pd.read_csv(file_path)

print("2. Tallying all email appearances...")
address_counts = Counter()
total_emails_processed = 0

for raw_message in df['message']:
    msg = email.message_from_string(raw_message)
    sender = msg.get('From')
    receivers_raw = msg.get('To')
    
    if sender:
        address_counts[sender.strip().lower()] += 1
    
    if receivers_raw:
        receivers = [r.strip().lower() for r in receivers_raw.replace('\n', '').split(',')]
        for r in receivers:
            if r:
                address_counts[r] += 1
                
    total_emails_processed += 1
    if total_emails_processed % 100000 == 0:
        print(f"   ...processed {total_emails_processed} emails")

print("3. Calculating Observation Bias...")
total_unique_addresses = len(address_counts)
total_mentions = sum(address_counts.values())

# The ~150 people whose mailboxes were actually dumped by the FBI/CMU
num_custodians = 150
top_150 = address_counts.most_common(num_custodians)

custodian_mentions = sum([count for addr, count in top_150])
non_custodian_mentions = total_mentions - custodian_mentions

# Calculate Percentages
custodian_pop_pct = (num_custodians / total_unique_addresses) * 100
non_custodian_pop_pct = 100 - custodian_pop_pct

custodian_vol_pct = (custodian_mentions / total_mentions) * 100
non_custodian_vol_pct = 100 - custodian_vol_pct

print(f"Total Unique Addresses: {total_unique_addresses}")
print(f"Top 150 Volume: {custodian_vol_pct:.1f}%")

# --- GENERATE THE POSTER GRAPHIC ---
print("4. Generating Graphic...")
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 6))

# Data setup
categories = ['The 150 Subpoenaed Custodians\n(e.g., skilling-j, lay-k)', 'All Other Employees & Externals\n(The "Unseen" Network)']
population_pcts = [custodian_pop_pct, non_custodian_pop_pct]
volume_pcts = [custodian_vol_pct, non_custodian_vol_pct]

x = np.arange(len(categories))
width = 0.35

rects1 = ax.bar(x - width/2, population_pcts, width, label='% of Total Network Population', color='#55A868')
rects2 = ax.bar(x + width/2, volume_pcts, width, label='% of Total Email Volume', color='#C44E52')

ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('The Observation Bias of the Enron Corpus', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.legend(fontsize=11)

# Attach a text label above each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# Add a descriptive text box to explain the phenomenon
textstr = "Data Collection Bias:\nBecause only 150 specific mailboxes (maildirs)\nwere collected, edges between non-custodians\nare mathematically impossible to observe,\nartificially inflating the centrality of the core 150."
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig("Poster_Chart_3_ObservationBias.png", dpi=300)
print("✅ Saved to 'Poster_Chart_3_ObservationBias.png'")
plt.show()