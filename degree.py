import networkx as nx
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # Suppress warnings for cleaner output

# Parameters
n = 500  # Number of nodes
max_degree = n - 1  # Maximum possible degree
min_degree = 1      # Minimum degree for connectivity
max_mean = 20       # Maximum mean degree to test
gcc_threshold = 0.9  # Giant component must cover ≥90% of nodes
path_threshold = 6.0  # Average shortest path length ≤6

# Function to generate degree sequence and create graph
def generate_graph(distribution, mean_degree):
    if distribution == "uniform":
        # Uniform distribution over [1, max_degree]
        degrees = np.random.randint(1, max_degree + 1, size=n)
        degrees = degrees * mean_degree / np.mean(degrees)  # Scale to desired mean
        degrees = degrees.astype(int)
    elif distribution == "normal":
        # Normal distribution, ensure degrees are integers ≥1
        degrees = np.random.normal(loc=mean_degree, scale=mean_degree/2, size=n)
        degrees = np.clip(degrees, 1, max_degree).astype(int)
    elif distribution == "exponential":
        # Exponential distribution
        degrees = np.random.exponential(scale=mean_degree, size=n)
        degrees = np.clip(degrees, 1, max_degree).astype(int)
    elif distribution == "powerlaw":
        # Power-law degree sequence using NetworkX
        degrees = nx.utils.powerlaw_sequence(n, exponent=2.5)
        degrees = np.clip(degrees, 1, max_degree).astype(int)
    else:
        raise ValueError("Unknown distribution")

    # Ensure sum of degrees is even
    if sum(degrees) % 2 != 0:
        degrees[0] += 1
    
    try:
        # Generate graph using configuration model
        G = nx.configuration_model(degrees)
        G = nx.Graph(G)  # Remove parallel edges and self-loops
        G.remove_edges_from(nx.selfloop_edges(G))
        return G
    except:
        return None

# Function to check if graph meets criteria
def check_graph(G):
    if G is None or len(G) == 0:
        return False
    # Find the largest connected component
    gcc = max(nx.connected_components(G), key=len, default=set())
    gcc_size = len(gcc)
    if gcc_size / n < gcc_threshold:
        return False
    # Compute average shortest path length for GCC
    gcc_subgraph = G.subgraph(gcc)
    try:
        avg_path_length = nx.average_shortest_path_length(gcc_subgraph)
        return avg_path_length <= path_threshold
    except:
        return False

# Find threshold mean degree for each distribution
distributions = ["uniform", "normal", "exponential", "powerlaw"]
results = {}

for dist in distributions:
    for mean_degree in range(min_degree, max_mean + 1):
        # Generate multiple graphs to account for randomness
        success = False
        for _ in range(5):  # Try 5 times to ensure robustness
            G = generate_graph(dist, mean_degree)
            if check_graph(G):
                success = True
                break
        if success:
            results[dist] = mean_degree
            break
    else:
        results[dist] = None  # No threshold found within max_mean

# Print results
print("Threshold average degree for each distribution:")
for dist, threshold in results.items():
    print(f"{dist.capitalize():<12}: {threshold if threshold else 'Not found within max mean degree'}")