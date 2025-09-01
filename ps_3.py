#PROGRAMMING ASSIGNMENT 3
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Generate power-law degree sequence
def generate_power_law_sequence(N, gamma):
    degrees = np.random.pareto(gamma - 1, N).astype(int) + 1  # Minimum degree 1
    if sum(degrees) % 2 != 0:
        degrees[np.random.randint(0, N)] += 1
    return degrees

# Simulate attack and track giant component size
def simulate_attack(G, criterion, fractions):
    G_copy = G.copy()
    N = G_copy.number_of_nodes()
    sizes = []
    
    if criterion == 'degree':
        nodes_sorted = sorted(G_copy.nodes(), key=lambda x: G_copy.degree(x), reverse=True)
    else:  # clustering
        clustering = nx.clustering(G_copy)
        nodes_sorted = sorted(G_copy.nodes(), key=lambda x: clustering[x], reverse=True)
    
    for f in fractions:
        num_remove = int(f * N)
        G_temp = G_copy.copy()
        G_temp.remove_nodes_from(nodes_sorted[:num_remove])
        if G_temp.number_of_nodes() == 0:
            sizes.append(0)
        else:
            largest_cc = max(nx.connected_components(G_temp), key=len, default=set())
            sizes.append(len(largest_cc) / N)
    
    return sizes

# Plot giant component sizes
def plot_giant_component(fractions, sizes_degree, sizes_clustering, title):
    plt.figure(figsize=(8, 6))
    plt.plot(fractions, sizes_degree, label='Degree Attack', marker='o')
    plt.plot(fractions, sizes_clustering, label='Clustering Attack', marker='s')
    plt.xlabel('Fraction of Nodes Removed')
    plt.ylabel('Giant Component Size (Normalized)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    N = 10000
    fractions = np.linspace(0, 0.5, 11)  # 0 to 50% in 5% steps

    # Task 1: Configuration Model (Power-law, γ = 2.5)
    degrees = generate_power_law_sequence(N, gamma=2.5)
    G_config = nx.configuration_model(degrees)
    G_config = nx.Graph(G_config)  # Convert to simple graph
    sizes_degree_config = simulate_attack(G_config, 'degree', fractions)
    sizes_clustering_config = simulate_attack(G_config, 'clustering', fractions)
    print("Configuration Model (γ = 2.5):")
    print(f"Degree Attack: Giant component sizes = {[f'{s:.3f}' for s in sizes_degree_config]}")
    print(f"Clustering Attack: Giant component sizes = {[f'{s:.3f}' for s in sizes_clustering_config]}")
    plot_giant_component(fractions, sizes_degree_config, sizes_clustering_config, 
                        'Giant Component Size vs Fraction Removed (Configuration Model, γ=2.5)')

    # Task 2: Hierarchical Model
    G_hierarchical = nx.powerlaw_cluster_graph(N, m=4, p=0.1)
    sizes_degree_hier = simulate_attack(G_hierarchical, 'degree', fractions)
    sizes_clustering_hier = simulate_attack(G_hierarchical, 'clustering', fractions)
    print("\nHierarchical Model (m=4, p=0.1):")
    print(f"Degree Attack: Giant component sizes = {[f'{s:.3f}' for s in sizes_degree_hier]}")
    print(f"Clustering Attack: Giant component sizes = {[f'{s:.3f}' for s in sizes_clustering_hier]}")
    plot_giant_component(fractions, sizes_degree_hier, sizes_clustering_hier, 
                        'Giant Component Size vs Fraction Removed (Hierarchical Model)')