from qaoa import Quantum_QAOA
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Create a graph using NetworkX
G = nx.Graph()  # Initialize an empty undirected graph
edges = np.array([(1, 2), (1, 3), (2,3), (3, 4), (3, 5),(4, 5), (5, 6)])  # Define edges as pairs of nodes
G.add_edges_from(edges)  # Add the defined edges to the graph


# Visualize the graph
nx.draw(G, with_labels = True) # Draw the graph with node labels
plt.show()  # Display the graph visualization


# Initialize the Quantum_QAOA class with the graph and 2 QAOA layers
test = Quantum_QAOA(G, layers=2)

# Print the register to visualize the qubit arrangement and blockade radius
test.print_reg()

# Run the QAOA algorithm and generate a histogram of results
test.run_with_optimization(G, generate_histogram=True)
