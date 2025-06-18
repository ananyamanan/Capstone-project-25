import torch
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx

# Load list of graph objects
graph_list = torch.load('relationship_graphs.pt')

# Check how many graphs you have
print(f"Total graphs: {len(graph_list)}")

# Select the first graph
graph_data = graph_list[0]  # or graph_list[i] for other indices

# Convert to NetworkX
G = to_networkx(graph_data, to_undirected=True)

# Draw the graph
plt.figure(figsize=(6, 6))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
plt.title("Graph 0 from relationship_graphs.pt")
plt.show()
