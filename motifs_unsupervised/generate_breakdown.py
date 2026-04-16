import re
import matplotlib.pyplot as plt
from collections import Counter

# 1. Parse the HTML File
file_path = "motifs_unsupervised\\MotifsUnsupervisedOutput\\motifs_unsupervised_20260410-0922.html"
with open(file_path, "r", encoding="utf-8") as f:
    html = f.read()

slices = re.split(r'<h2>📅 Time Slice: (.+?)</h2>', html)[1:]

pure_clusters = Counter()
mixed_clusters_count = 0
total_vips_in_pure = 0
total_vips_in_mixed = 0

# 2. Categorize the Clusters
for i in range(0, len(slices), 2):
    content = slices[i+1]
    clusters = re.findall(r'<div class="cluster-box">.*?<div class="vip-list">⭐ VIPs Found: (.*?)</div>', content, re.DOTALL)
    
    for cluster_vips_text in clusters:
        if '(None)' not in cluster_vips_text:
            roles = []
            for item in cluster_vips_text.split(', '):
                match = re.match(r'(.+) \((.+)\)', item.strip())
                if match:
                    roles.append(match.group(2))
            
            # Check Purity
            unique_roles = set(roles)
            if len(unique_roles) == 1:
                # It's a pure cluster!
                role = list(unique_roles)[0]
                pure_clusters[role] += 1
                total_vips_in_pure += len(roles)
            elif len(unique_roles) > 1:
                # It's a mixed/failed cluster
                mixed_clusters_count += 1
                total_vips_in_mixed += len(roles)

# 3. Prepare Data for the Chart
labels = ['Mixed Roles\n(Algorithm Confusion)']
sizes = [mixed_clusters_count]
colors = ['#b0b0b0'] # Dull gray for the "noise"
explode = [0.05] # Slightly detach the noise

# Vibrant colors for the successes
success_colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974'] 

color_idx = 0
for role, count in pure_clusters.items():
    labels.append(f'Pure: {role}\n(Algorithm Success)')
    sizes.append(count)
    colors.append(success_colors[color_idx % len(success_colors)])
    explode.append(0.0)
    color_idx += 1

# 4. Generate the Donut Chart
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 8))

# Create the pie chart
wedges, texts, autotexts = ax.pie(
    sizes, 
    labels=labels, 
    colors=colors, 
    explode=explode,
    autopct='%1.1f%%', 
    startangle=140,
    textprops={'fontsize': 12, 'fontweight': 'bold'}
)

# Draw a white circle in the middle to make it a Donut Chart
centre_circle = plt.Circle((0,0), 0.60, fc='white')
fig.gca().add_artist(centre_circle)

# Add a text label in the center
center_text = f"Total VIP\nClusters\n{sum(sizes)}"
ax.text(0, 0, center_text, ha='center', va='center', fontsize=20, fontweight='bold', color='#333')

plt.title('Unsupervised Motif Accuracy', fontsize=18, fontweight='bold', pad=20)

plt.tight_layout()
output_name = "Poster_Chart_5_PurityBreakdown.png"
plt.savefig(output_name, dpi=300)
print(f"✅ Saved graphic to {output_name}")

# Print out some "Wins" for you to highlight in text on the poster
print("\n--- Highlight these specific successes on your poster! ---")
print(f"Total perfectly pure clusters found: {sum(pure_clusters.values())}")
print(f"Total mixed (confused) clusters found: {mixed_clusters_count}")
for role, count in pure_clusters.items():
    print(f"- The model perfectly isolated '{role}' {count} times.")