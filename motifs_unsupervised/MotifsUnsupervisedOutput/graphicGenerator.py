import re
import matplotlib.pyplot as plt
import numpy as np

def parse_html_regex(filepath):
    """Reads the HTML report and extracts VIP roles from each cluster."""
    with open(filepath, "r", encoding="utf-8") as f:
        html = f.read()
        
    slices = re.split(r'<h2>📅 Time Slice: (.+?)</h2>', html)[1:]
    data = {}
    
    for i in range(0, len(slices), 2):
        slice_name = slices[i]
        content = slices[i+1]
        
        # Find every cluster box and extract the VIP line
        clusters = re.findall(r'<div class="cluster-box">.*?<div class="vip-list">⭐ VIPs Found: (.*?)</div>', content, re.DOTALL)
        
        parsed_clusters = []
        for cluster_vips_text in clusters:
            vips = []
            if '(None)' not in cluster_vips_text:
                # Split by comma to get individual VIPs
                for item in cluster_vips_text.split(', '):
                    match = re.match(r'(.+) \((.+)\)', item.strip())
                    if match:
                        vips.append((match.group(1), match.group(2)))
            parsed_clusters.append(vips)
        data[slice_name] = parsed_clusters
    return data

def calculate_purity(data):
    """
    Calculates overall Cluster Purity:
    Sum of the majority class in each cluster divided by total VIPs.
    """
    total_vips = 0
    majority_sum = 0
    
    for t_slice, clusters in data.items():
        for cluster in clusters:
            if len(cluster) > 0:
                total_vips += len(cluster)
                
                # Count the frequency of each role in this specific cluster
                role_counts = {}
                for name, role in cluster:
                    role_counts[role] = role_counts.get(role, 0) + 1
                    
                # Find the role that appears the most and add its count
                majority_sum += max(role_counts.values())
                
    return (majority_sum / total_vips) * 100 if total_vips > 0 else 0

# ==========================================
# 1. LOAD DATA & CALCULATE METRICS
# ==========================================
# Update these filenames if necessary!
base_file = "motifs_unsupervised_20260410-0922.html"
nlp_file = "motifs_unsupervised_nlp_20260412-0909.html"

# Plug in your estimated run times here (in minutes)
time_base_minutes = 5.0 
time_nlp_minutes = 65.0 

data_base = parse_html_regex(base_file)
data_nlp = parse_html_regex(nlp_file)

purity_base = calculate_purity(data_base)
purity_nlp = calculate_purity(data_nlp)

print(f"--- RESULTS ---")
print(f"Base Topology Model Purity: {purity_base:.2f}%")
print(f"NLP Enhanced Model Purity:  {purity_nlp:.2f}%\n")

# ==========================================
# 2. GENERATE POSTER CHARTS
# ==========================================
plt.style.use('ggplot')

# Chart A: Purity Bar Chart
fig, ax = plt.subplots(figsize=(8, 6))
labels = ['Pure Topology', 'Topology + NLP']
scores = [purity_base, purity_nlp]
colors = ['#4C72B0', '#C44E52']

bars = ax.bar(labels, scores, color=colors, width=0.5)
ax.set_ylabel('Cluster Purity (%)', fontsize=12, fontweight='bold')
ax.set_title('Corporate Role Isolation Accuracy (Occam\'s Razor)', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(scores) + 10)

# Add exact numbers on top of bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("Poster_Chart_1_Purity.png", dpi=300)
print("Saved: Poster_Chart_1_Purity.png")

# Chart B: Efficiency Scatter Plot (Cost vs. Benefit)
fig2, ax2 = plt.subplots(figsize=(8, 6))

times = [time_base_minutes, time_nlp_minutes]
ax2.scatter(times, scores, s=300, c=colors, edgecolors='black', zorder=5)

ax2.set_xlabel('Execution Time (Minutes)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cluster Purity (%)', fontsize=12, fontweight='bold')
ax2.set_title('Algorithm Efficiency: Feature Washout Effect', fontsize=14, fontweight='bold')

# Annotate points
ax2.annotate('Pure Topology\n(High Purity, Fast)', 
             xy=(times[0], scores[0]), xytext=(times[0]+2, scores[0]),
             fontsize=11, fontweight='bold', color=colors[0])

ax2.annotate('Topology + NLP\n(Lower Purity, Slow)', 
             xy=(times[1], scores[1]), xytext=(times[1]-15, scores[1]-1.5),
             fontsize=11, fontweight='bold', color=colors[1])

# Adjust limits for visual padding
ax2.set_xlim(0, max(times) + 10)
ax2.set_ylim(min(scores) - 5, max(scores) + 5)
plt.grid(True, linestyle='--', alpha=0.7, zorder=0)

plt.tight_layout()
plt.savefig("Poster_Chart_2_Efficiency.png", dpi=300)
print("Saved: Poster_Chart_2_Efficiency.png")