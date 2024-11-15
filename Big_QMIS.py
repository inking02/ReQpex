import numpy as np
import networkx as nx
from QMIS_code.Quantum_MIS import Quantum_MIS
from utils.utils import disc_graph_to_connected
from numpy.typing import NDArray
from sklearn.cluster import KMeans
import pymetis


# À voir si on fait juste un graph en entree
class BIG_QMIS:
    def __init__(self, graph: nx.Graph, num_atoms: int = 15) -> None:
        self.graph = graph
        if num_atoms > 20:
            self.num_atoms = 20
        else:
            self.num_atoms = num_atoms

    def run(self):
        num_of_cuts = int(np.ceil(self.graph.number_of_nodes() / self.num_atoms))
        adjacency_list = [
            list(map(int, list(self.graph.neighbors(node))))
            for node in self.graph.nodes()
        ]
        print(adjacency_list)
        n_cuts, membership = pymetis.part_graph(num_of_cuts, adjacency=adjacency_list)
        # n_cuts = 3
        # membership = [1, 1, 1, 0, 1, 0, 0]

        nodes = []
        for i in range(num_of_cuts):
            nodes.append(np.argwhere(np.array(membership) == i).ravel())
        print(nodes)
        print("Nombres de sommets par sous-graphe")
        for i, l in enumerate(nodes):
            print("Taille ensemble ", i, " = ", np.shape(l))
