#PROGRAMMING AASIGNMENT 1

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Task 1: Generate and visualize G(N, p) networks
def generate_and_visualize_erdos_renyi(N, avg_degree, title):
    p = avg_degree / (N - 1)
    G = nx.erdos_renyi_graph(N, p)
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_size=20, node_color='blue', with_labels=False)
    plt.title(title)
    plt.show()  # Display the plot

# Task 2: Red and Blue network
def generate_red_blue_network(N, p, q):
    G = nx.Graph()
    nodes = list(range(2 * N))
    G.add_nodes_from(nodes)
    colors = {i: 'red' if i < N else 'blue' for i in nodes}
    nx.set_node_attributes(G, colors, 'color')
    for i in range(2 * N):
        for j in range(i + 1, 2 * N):
            if colors[i] == colors[j] and np.random.random() < p:
                G.add_edge(i, j)
            elif colors[i] != colors[j] and np.random.random() < q:
                G.add_edge(i, j)
    return G

def check_connectivity(G):
    return nx.number_connected_components(G)

def average_shortest_path_length(G):
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    return float('inf')

# Task 3: Red, Blue, Purple network
def generate_red_blue_purple_network(N, f, p):
    total_nodes = 2 * N
    num_purple = int(f * total_nodes)
    num_red_blue = total_nodes - num_purple
    N_red = num_red_blue // 2
    N_blue = num_red_blue - N_red
    G = nx.Graph()
    nodes = list(range(total_nodes))
    G.add_nodes_from(nodes)
    colors = {i: 'red' if i < N_red else 'blue' if i < N_red + N_blue else 'purple' for i in nodes}
    nx.set_node_attributes(G, colors, 'color')
    for i in range(total_nodes):
        for j in range(i + 1, total_nodes):
            if colors[i] == colors[j] and colors[i] in ['red', 'blue'] and np.random.random() < p:
                G.add_edge(i, j)
            elif colors[i] == 'purple' and colors[j] in ['red', 'blue'] and np.random.random() < p:
                G.add_edge(i, j)
            elif colors[j] == 'purple' and colors[i] in ['red', 'blue'] and np.random.random() < p:
                G.add_edge(i, j)
    return G

def check_two_step_connectivity(G, N):
    red_nodes = [n for n, attr in G.nodes(data=True) if attr['color'] == 'red']
    blue_nodes = [n for n, attr in G.nodes(data=True) if attr['color'] == 'blue']
    for red in red_nodes[:10]:  # Sample 10 nodes
        for blue in blue_nodes[:10]:
            paths = list(nx.all_simple_paths(G, red, blue, cutoff=2))
            if not any(len(path) == 3 for path in paths):
                return False
    return True

# Main execution
if __name__ == "__main__":
    N = 500

    # Task 1: Generate and display three networks
    avg_degrees = [0.8, 1, 8]
    for k in avg_degrees:
        title = f'Erdős-Rényi Network (N={N}, <k>={k})'
        generate_and_visualize_erdos_renyi(N, k, title)

    # Task 2b: Check connectivity
    p = 1 / (N - 1)
    q = 1 / N
    G_rb = generate_red_blue_network(N, p, q)
    num_components = check_connectivity(G_rb)
    print(f"Task 2b: Number of components with p={p:.6f}, q={q:.6f}: {num_components}")

    # Task 2c: Small-world property
    p_snobbish = 0.01
    q_snobbish = 0.0001
    G_snobbish = generate_red_blue_network(N, p_snobbish, q_snobbish)
    if nx.is_connected(G_snobbish):
        avg_path = average_shortest_path_length(G_snobbish)
        print(f"Task 2c: Average shortest path length (p={p_snobbish}, q={q_snobbish}): {avg_path:.2f}")
    else:
        print("Task 2c: Network is not connected")

    # Task 3a: Test two-step connectivity
    p = 8 / (N - 1)
    f_values = [0.01, 0.05, 0.1]
    for f in f_values:
        G_rbp = generate_red_blue_purple_network(N, f, p)
        is_interactive = check_two_step_connectivity(G_rbp, N)
        print(f"Task 3a: f={f}, Interactive: {is_interactive}")