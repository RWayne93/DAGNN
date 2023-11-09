import json
import matplotlib.pyplot as plt
import numpy as np

# Load your data from 'data.json'
with open('network_states.json', 'r') as f:
    data = json.load(f)

# Filter the data by score of 64
score_filtered_data = {k: v for k, v in data.items() if v['score'] == 64}

# Collect the network sizes
sizes = [details['size'] for details in score_filtered_data.values()]

# Calculate the mean size for the runs with score of 64
mean_size = np.mean(sizes)

# Count the occurrences of each network size
size_counts = {}
for size in sizes:
    if size in size_counts:
        size_counts[size] += 1
    else:
        size_counts[size] = 1

# Sort the sizes for plotting
sorted_sizes = sorted(size_counts.keys())
counts = [size_counts[size] for size in sorted_sizes]

# Plotting
plt.figure(figsize=(10, 6))

# Create a bar plot
plt.bar(sorted_sizes, counts, color='purple')

# Indicate the mean size with a dashed line
plt.axvline(x=mean_size, color='red', linestyle='--', label=f'Mean Size: {mean_size:.2f}')

# Adding labels and title
plt.xlabel('Network Size')
plt.ylabel('Count')
plt.title('Distribution of Network Sizes with Score 64')
plt.legend()

plt.show()
