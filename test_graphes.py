import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Big_QMIS import BIG_QMIS


G = nx.Graph()  # Création d'un graphe sans sommet et sans arête
nodes = [0, 1, 2, 3, 4]
nodes = [str(i) for i in nodes]
edges = [(0, 1), (0, 2), (0, 4), (1, 2), (1, 4), (1, 3), (2, 4), (2, 3)]
edges = [tuple([str(j), str(k)]) for (j, k) in edges]
G.add_nodes_from(nodes)  # On ajoute 3 sommets
G.add_edges_from(edges)
nx.draw(G, with_labels=True)

solver = BIG_QMIS(G, num_atoms=3)
print(solver.run(print_progression=True))
