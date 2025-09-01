#Programming Assignment 4
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Generate power-law degree sequence for scale-free network
def generate_scale_free_degree_sequence(N, gamma, avg_degree):
    degrees = np.random.pareto(gamma - 1, N) * N ** (1 / (gamma - 1))
    degrees = degrees.astype(int) + 1  # Ensure positive integers
    current_avg = np.mean(degrees)
    degrees = (degrees * (avg_degree / current_avg)).astype(int)
    if sum(degrees) % 2 != 0:
        degrees[np.random.randint(0, N)] += 1
    return degrees

# Simulate sandpile model
def simulate_sandpile(G, steps=10):
    buckets = {node: G.degree(node) for node in G.nodes()}
    grains = {node: 0 for node in G.nodes()}
    avalanche_sizes = []
    
    for _ in range(steps):
        node = np.random.choice(list(G.nodes()))
        grains[node] += 1
        
        unstable = deque([node]) if grains[node] >= buckets[node] else deque()
        current_avalanche_size = 0
        
        while unstable:
            current_node = unstable.popleft()
            if grains[current_node] >= buckets[current_node]:
                current_avalanche_size += 1
                num_grains = grains[current_node]
                grains[current_node] = 0
                neighbors = list(G.neighbors(current_node))
                if neighbors:
                    grains_per_neighbor = num_grains // len(neighbors)
                    for neighbor in neighbors:
                        grains[neighbor] += grains_per_neighbor
                        if grains[neighbor] >= buckets[neighbor]:
                            unstable.append(neighbor)
        
        if current_avalanche_size > 0:
            avalanche_sizes.append(current_avalanche_size)
    
    return avalanche_sizes

# Plot avalanche size distribution
def plot_avalanche_distribution(avalanche_sizes, title):
    plt.figure(figsize=(8, 6))
    plt.hist(avalanche_sizes, bins=30, log=True, density=True)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Avalanche Size')
    plt.ylabel('Frequency (Log Scale)')
    plt.title(title)
    plt.grid(True, which="both", ls="--")
    plt.show()

# Main execution
if __name__ == "__main__":
    N = 50  # Reduced from 1000
    avg_degree = 2
    
    # Erdős-Rényi network
    p = avg_degree / (N - 1)
    G_er = nx.erdos_renyi_graph(N, p)
    avalanche_sizes_er = simulate_sandpile(G_er, steps=1000)  # Reduced steps
    print(f"Erdős-Rényi: Mean avalanche size = {np.mean(avalanche_sizes_er):.2f}")
    plot_avalanche_distribution(avalanche_sizes_er, 
                              f'Sandpile Avalanche Distribution (Erdős-Rényi, N={N}, <k>={avg_degree})')
    
    # Scale-free network
    degrees = generate_scale_free_degree_sequence(N, gamma=2.5, avg_degree=avg_degree)
    G_sf = nx.configuration_model(degrees)
    G_sf = nx.Graph(G_sf)  # Convert to simple graph
    avalanche_sizes_sf = simulate_sandpile(G_sf, steps=10)  # Reduced steps
    print(f"Scale-Free: Mean avalanche size = {np.mean(avalanche_sizes_sf):.2f}")
    plot_avalanche_distribution(avalanche_sizes_sf, 
                              f'Sandpile Avalanche Distribution (Scale-Free, N={N}, <k>={avg_degree})')