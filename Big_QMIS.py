import numpy as np
import networkx as nx
from QMIS_code.Quantum_MIS import Quantum_MIS
from numpy.typing import NDArray
from typing import List
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
            list(
                map(int, list(self.graph.neighbors(node)))
            )  # À améliorer sinon ce n'est pas des int mais des str
            for node in self.graph.nodes()
        ]
        n_cuts, membership = pymetis.part_graph(num_of_cuts, adjacency=adjacency_list)
        # n_cuts = 3
        # membership = [1, 1, 1, 0, 1, 0, 0]

        sub_graphs = []
        for i in range(num_of_cuts):
            nodes = np.argwhere(np.array(membership) == i).ravel()
            sub_graphs.append(self.create_sub_graph(nodes))

        MIS_list = np.empty_like(sub_graphs)

        for i, graph in enumerate(sub_graphs):
            MIS_object = Quantum_MIS(graph)  # À faire
            res_dict = MIS_object.run(shots=100)
            MIS_list[i] = max(zip(res_dict.values(), res_dict.keys()))[1]

    def create_sub_graph(self, nodes: List[int]):
        # Créer manuellement avec chat, trouver pourquoi la fonction subgraph marche pas
        subgraph = nx.Graph()
        subgraph.add_nodes_from(nodes)
        subgraph.add_edges_from(
            (u, v) for u, v in self.graph.edges(nodes) if u in nodes and v in nodes
        )
        return subgraph
