from Quantum_MIS import Quantum_MIS
import numpy as np
import networkx as nx

G= nx.Graph()
edges = np.array([(1, 2), (1, 3), (2,3), (3, 4), (3, 5),(4, 5), (5, 6)])
G.add_edges_from(edges)
pos = nx.spring_layout(G, k = 1,   seed = 42)


test =  Quantum_MIS(np.array(list(pos.values())))
test.print_reg()

test.run(generate_histogram=True)