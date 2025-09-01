#sample code with GML file

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Function to compute network metrics
def compute_metrics(G, name):
    # Average clustering coefficient
    avg_clustering = nx.average_clustering(G)
    
    # Average shortest path length (for the largest connected component)
    if nx.is_directed(G):
        G_cc = max(nx.strongly_connected_components(G), key=len)
        G_sub = G.subgraph(G_cc).copy()
    else:
        G_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(G_cc).copy()
    avg_path_length = nx.average_shortest_path_length(G_sub)
    
    # Average degree
    degrees = [d for n, d in G.degree()]
    avg_degree = np.mean(degrees)
    
    print(f"\nMetrics for {name}:")
    print(f"Average Clustering Coefficient: {avg_clustering:.4f}")
    print(f"Average Shortest Path Length: {avg_path_length:.4f}")
    print(f"Average Degree: {avg_degree:.4f}")
    return avg_clustering, avg_path_length, avg_degree, degrees

# Function to visualize network and degree distribution
def visualize_network(G, name, degrees):
    # Create figure with two subplots: network layout and degree distribution
    plt.figure(figsize=(12, 5))
    
    # Network layout (spring layout)
    plt.subplot(121)
    pos = nx.spring_layout(G, seed=42)  # Consistent layout for reproducibility
    nx.draw(G, pos, node_size=50, node_color='skyblue', edge_color='gray', with_labels=False)
    plt.title(f"{name} Layout")
    
    # Degree distribution histogram
    plt.subplot(122)
    plt.hist(degrees, bins=20, density=True, color='salmon', edgecolor='black')
    plt.title(f"{name} Degree Distribution")
    plt.xlabel('Degree')
    plt.ylabel('Probability')
    
    plt.tight_layout()
    plt.show()

# Load the dataset from a GML file
try:
    G_data = nx.read_gml('network.gml')
    print("Loaded network from GML file")
    avg_clust, avg_path, avg_deg, degrees_data = compute_metrics(G_data, "Input Network")
    visualize_network(G_data, "Input Network", degrees_data)
except FileNotFoundError:
    print("GML file not found. Please provide a valid GML file path.")
    # Fallback: Create a sample graph for demonstration
    G_data = nx.karate_club_graph()
    print("Using Karate Club graph as fallback")
    avg_clust, avg_path, avg_deg, degrees_data = compute_metrics(G_data, "Karate Club Network")
    visualize_network(G_data, "Karate Club Network", degrees_data)

# Number of nodes for generated networks
n = G_data.number_of_nodes()
# Approximate number of edges for realistic comparisons
m = G_data.number_of_edges()
# Estimate edge probability for Erdős-Rényi
p = (2 * m) / (n * (n - 1)) if not nx.is_directed(G_data) else m / (n * (n - 1))

# 1. Erdős-Rényi Graph
G_er = nx.erdos_renyi_graph(n, p)
avg_clust_er, avg_path_er, avg_deg_er, degrees_er = compute_metrics(G_er, "Erdős-Rényi Graph")
visualize_network(G_er, "Erdős-Rényi Graph", degrees_er)

# 2. Watts-Strogatz Model
k = int(np.mean([d for n, d in G_data.degree()]))
G_ws = nx.watts_strogatz_graph(n, k, 0.1)
avg_clust_ws, avg_path_ws, avg_deg_ws, degrees_ws = compute_metrics(G_ws, "Watts-Strogatz Graph")
visualize_network(G_ws, "Watts-Strogatz Graph", degrees_ws)

# 3. Scale-Free Network (Barabási-Albert model)
m_ba = max(1, int(m / n)) # Ensure at least 1 edge
G_sf = nx.barabasi_albert_graph(n, m_ba)
avg_clust_sf, avg_path_sf, avg_deg_sf, degrees_sf = compute_metrics(G_sf, "Scale-Free Network")
visualize_network(G_sf, "Scale-Free Network", degrees_sf)
