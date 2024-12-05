import networkx as nx
import matplotlib.pyplot as plt
from Big_QMIS import BIG_QMIS


n_nodes = 15
p_edge = 0.1  # Faible probabilité pour que le graphe ne soit pas trop dense
graph = nx.erdos_renyi_graph(n=n_nodes, p=p_edge, seed=42)
pos = nx.spring_layout(graph, k = 0.1, seed = 42)

nx.draw(graph, with_labels=True, node_color="lightblue", node_size=500, edge_color="gray", pos = pos)
plt.show()

solver = BIG_QMIS(graph)
r = solver.run()


colored_nodes = [int(value) for value in r]
print(colored_nodes)

node_colors = ["red" if node in colored_nodes else "lightblue" for node in graph.nodes]
nx.draw(graph, with_labels=True, node_color=node_colors, node_size=500, edge_color="gray", pos = pos)
plt.show()