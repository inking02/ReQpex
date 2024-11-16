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
        nodes_per_graph = []
        for i in range(num_of_cuts):
            nodes = np.argwhere(np.array(membership) == i).ravel()
            nodes_per_graph.append(nodes)
            sub_graphs.append(self.create_sub_graph(nodes))

        MIS_list = np.empty_like(sub_graphs)

        for i, (graph, nodes) in enumerate(zip(sub_graphs, nodes_per_graph)):
            MIS_object = Quantum_MIS(graph)  # À faire
            res_dict = MIS_object.run(shots=100)
            best_bitstring = max(zip(res_dict.values(), res_dict.keys()))[1]
            independant_nodes = []
            for j in range(len(best_bitstring)):
                if best_bitstring[j] == "1":
                    independant_nodes.append((nodes_per_graph[i])[j])
            MIS_list[i] = independant_nodes

        return self.combine_mis(MIS_list)
        # À vérifier ce bloc de for quand fonction de quantique sera à jour.

    def create_sub_graph(self, nodes: List[int]):
        # Créer manuellement avec chat, trouver pourquoi la fonction subgraph marche pas
        subgraph = nx.Graph()
        subgraph.add_nodes_from(nodes)
        subgraph.add_edges_from(
            (u, v) for u, v in self.graph.edges(nodes) if u in nodes and v in nodes
        )
        return subgraph

    def combine_mis(self, MIS_list: List[List[int]]):
        if len(MIS_list) == 1:
            return MIS_list
        n = len(MIS_list) // 2
        MIS_one = self.combine_mis(MIS_list[:n])
        MIS_two = self.combine_mis(MIS_list[n:])

        # À vérifier mais devrait fonctionner
        forest = nx.Graph()
        forest.add_nodes_from(MIS_one)
        forest.add_nodes_from(MIS_two)
        forest.add_edges_from(
            (u, v)
            for u, v in self.graph.edges(forest.nodes())
            if u in MIS_one and v in MIS_two
        )

        new_mis = [nx.isolate(forest)]
        forest.remove_edges_from(
            nx.isolate(new_mis)
        )  # Devrait marcher pour enlever nodes pas impliqués dans la combinaison

        for tree in nx.connected_components(forest):
            pass

        return new_mis
