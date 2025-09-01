
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.community import girvan_newman
import itertools

# Set random seed for reproducibility
np.random.seed(42)

### 1. Graph Creation and Modification
def create_basic_graph():
    """Create and modify a simple graph."""
    G = nx.Graph()  # Undirected graph (use nx.DiGraph() for directed)
    G.add_nodes_from([1, 2, 3, 4])  # Add nodes
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])  # Add edges
    G.add_weighted_edges_from([(1, 2, 1.5), (2, 3, 2.0)])  # Add weighted edges
    # Remove nodes/edges
    # G.remove_node(1)
    # G.remove_edges_from([(1, 2)])
    return G

### 2. Generate Standard and Random Graphs
def generate_complete_graph(n):
    """Generate a complete graph with n nodes."""
    return nx.complete_graph(n)

def generate_cycle_graph(n):
    """Generate a cycle graph with n nodes."""
    return nx.cycle_graph(n)

def generate_erdos_renyi(n, p):
    """Generate an Erdős-Rényi random graph with n nodes and edge probability p."""
    return nx.erdos_renyi_graph(n, p)

def generate_barabasi_albert(n, m):
    """Generate a Barabási-Albert scale-free graph with n nodes and m attachments."""
    return nx.barabasi_albert_graph(n, m)

def generate_configuration_model(N, gamma, m):
    """Generate a configuration model with power-law degree distribution."""
    degrees = [int(np.random.pareto(gamma - 1) * m) for _ in range(N)]
    if sum(degrees) % 2 != 0:
        degrees[0] += 1  # Ensure even sum
    try:
        G = nx.configuration_model(degrees)
        G = nx.Graph(G)  # Convert to simple graph
        G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops
        return G
    except:
        return generate_configuration_model(N, gamma, m)  # Retry if invalid

def generate_hierarchical_model(N):
    """Generate a simplified hierarchical model with high clustering."""
    G = nx.Graph()
    clique_size = 5
    G.add_nodes_from(range(clique_size))
    G.add_edges_from([(i, j) for i in range(clique_size) for j in range(i + 1, clique_size)])
    current_nodes = clique_size
    while current_nodes < N:
        G.add_node(current_nodes)
        targets = np.random.choice(list(G.nodes())[:current_nodes], size=min(4, current_nodes), replace=False)
        G.add_edges_from([(current_nodes, t) for t in targets])
        current_nodes += 1
    return G

### 3. Basic Graph Properties
def get_basic_properties(G):
    """Return basic properties of the graph."""
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    degrees = dict(G.degree())  # For DiGraph, use G.in_degree() or G.out_degree()
    neighbors = {node: list(G.neighbors(node)) for node in G.nodes()}
    return {
        'nodes': num_nodes,
        'edges': num_edges,
        'degrees': degrees,
        'neighbors': neighbors
    }

### 4. Network Metrics
def compute_metrics(G):
    """Compute common network metrics."""
    clustering = nx.clustering(G, weight='weight' if G.number_of_edges() > 0 and 'weight' in list(G.edges(data=True))[0] else None)
    avg_clustering = nx.average_clustering(G)
    degree_centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, weight='weight' if G.number_of_edges() > 0 and 'weight' in list(G.edges(data=True))[0] else None)
    pagerank = nx.pagerank(G, weight='weight' if G.number_of_edges() > 0 and 'weight' in list(G.edges(data=True))[0] else None)
    try:
        avg_shortest_path = nx.average_shortest_path_length(G)
    except:
        avg_shortest_path = None  # Handle disconnected graphs
    return {
        'clustering': clustering,
        'avg_clustering': avg_clustering,
        'degree_centrality': degree_centrality,
        'betweenness': betweenness,
        'pagerank': pagerank,
        'avg_shortest_path': avg_shortest_path
    }

def compute_communities(G, k=2):
    """Detect communities using Girvan-Newman algorithm."""
    communities = girvan_newman(G)
    return next(itertools.islice(communities, k-1, None))

def compute_shortest_paths(G, source):
    """Compute shortest paths from a source node."""
    return dict(nx.shortest_path_length(G, source=source, weight='weight' if G.number_of_edges() > 0 and 'weight' in list(G.edges(data=True))[0] else None))

### 5. Simulate Network Attacks
def simulate_attack(G, criterion, fractions):
    """Simulate node removal attack and track giant component size."""
    G_copy = G.copy()
    sizes = []
    total_nodes = G_copy.number_of_nodes()
    if criterion == 'degree':
        scores = dict(G_copy.degree())
    elif criterion == 'clustering':
        scores = nx.clustering(G_copy)
    else:
        raise ValueError("Criterion must be 'degree' or 'clustering'")
    
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    nodes_to_remove = [node for node, _ in sorted_nodes]
    
    for f in fractions:
        num_remove = int(f * total_nodes)
        G_temp = G_copy.copy()
        G_temp.remove_nodes_from(nodes_to_remove[:num_remove])
        largest_cc = max(nx.connected_components(G_temp), key=len, default=set())
        sizes.append(len(largest_cc) / total_nodes)
    return sizes

### 6. Visualization
def visualize_graph(G, node_sizes=None, node_colors=None, edge_widths=None, title="Graph Visualization"):
    """Visualize the graph with customizable attributes."""
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)  # Alternatives: nx.circular_layout, nx.kamada_kawai_layout
    node_sizes = node_sizes or [500] * G.number_of_nodes()
    node_colors = node_colors or 'lightblue'
    edge_widths = edge_widths or [1] * G.number_of_edges()
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color=node_colors,
            font_size=12, font_weight='bold', edge_color='gray', width=edge_widths)
    plt.title(title)
    plt.show()

### 7. Example Exam Scenario
def exam_example():
    """Solve a sample exam question: Create a Barabási-Albert graph, compute metrics, and visualize."""
    # Generate graph
    G = generate_barabasi_albert(100, 2)
    
    # Compute metrics
    degrees = dict(G.degree())
    clustering = nx.clustering(G)
    top_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    top_clustering = sorted(clustering.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("Top 5 nodes by degree:", top_degree)
    print("Top 5 nodes by clustering:", top_clustering)
    
    # Visualize with node sizes proportional to degree
    node_sizes = [degrees[node] * 100 for node in G.nodes()]
    visualize_graph(G, node_sizes=node_sizes, title="Barabási-Albert Graph (Node Size by Degree)")

### 8. Simulate Big Brother Attack (from previous query)
def big_brother_simulation():
    """Simulate degree and clustering attacks on configuration and hierarchical models."""
    N = 10000
    gamma = 2.5
    m = 5
    fractions = np.linspace(0, 0.5, 20)
    
    # Generate networks
    config_G = generate_configuration_model(N, gamma, m)
    hierarchical_G = generate_hierarchical_model(N)
    
    # Simulate attacks
    config_degree_sizes = simulate_attack(config_G, 'degree', fractions)
    config_clustering_sizes = simulate_attack(config_G, 'clustering', fractions)
    hierarchical_degree_sizes = simulate_attack(hierarchical_G, 'degree', fractions)
    hierarchical_clustering_sizes = simulate_attack(hierarchical_G, 'clustering', fractions)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(fractions, config_degree_sizes, label='Config Model (Degree Attack)', color='blue', linestyle='-')
    plt.plot(fractions, config_clustering_sizes, label='Config Model (Clustering Attack)', color='blue', linestyle='--')
    plt.plot(fractions, hierarchical_degree_sizes, label='Hierarchical Model (Degree Attack)', color='red', linestyle='-')
    plt.plot(fractions, hierarchical_clustering_sizes, label='Hierarchical Model (Clustering Attack)', color='red', linestyle='--')
    plt.xlabel('Fraction of Nodes Removed (f)')
    plt.ylabel('Giant Component Size (Relative to N)')
    plt.title('Giant Component Size Under Degree and Clustering Attacks')
    plt.legend()
    plt.grid(True)
    plt.show()

### 9. Main Function to Run Examples
if __name__ == "__main__":
    # Example: Create and visualize a simple graph
    G = create_basic_graph()
    print("Basic Graph Properties:", get_basic_properties(G))
    visualize_graph(G, title="Simple Graph")
    
    # Example: Compute metrics for a random graph
    G_er = generate_erdos_renyi(50, 0.1)
    metrics = compute_metrics(G_er)
    print("Erdős-Rényi Metrics:", {k: {n: round(v, 3) for n, v in metrics[k].items()} if isinstance(metrics[k], dict) else round(metrics[k], 3) for k in metrics})
    
    # Example: Community detection
    communities = compute_communities(G_er)
    print("Communities:", communities)
    
    # Run exam example
    exam_example()
    
    