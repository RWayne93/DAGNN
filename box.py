import matplotlib.pyplot as plt
import json
filepath = 'network_states.json'

# Load data
with open(filepath, 'r') as file:
    all_data = json.load(file)
    
# Process data
aggregated_data = {}

# Loop over the data and aggregate weights
for run_number, run_data in all_data.items():
    state = run_data['state']
    for node, attributes in state.items():
        links = attributes.get('links', {})
        for linked_node, weight in links.items():
            pair = f'{node}-{linked_node}'
            if pair not in aggregated_data:
                aggregated_data[pair] = []
            aggregated_data[pair] += [weight]

# Sort and prepare the scatter data
scatter_data = {'Node Pair': [], 'Weight': [], 'Y-Value': []}
for i, (node_pair, weights) in enumerate(sorted(aggregated_data.items(), key=lambda x: x[0])):
    for weight in weights:
        scatter_data['Node Pair'].append(node_pair)
        scatter_data['Weight'].append(weight)
        scatter_data['Y-Value'].append(i)

# Create scatter plot with increased opacity
plt.figure(figsize=(10, 8))
plt.scatter(scatter_data['Weight'], scatter_data['Y-Value'], alpha=0.1)  # alpha set to 1.0 for full opacity

# Set y-axis to have string pair names
plt.yticks(range(len(aggregated_data)), [x[0] for x in sorted(aggregated_data.items(), key=lambda x: x[0])])

# Set labels and title
plt.xlabel('Weight')
plt.ylabel('Node Pair')
plt.title('Scatter Plot of Weights by Node Pair')

# Display grid
plt.grid(True)

# Show the plot
plt.show()
