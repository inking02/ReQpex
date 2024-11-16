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

    def run(self, print_progression: bool = False):
        num_of_cuts = int(np.ceil(self.graph.number_of_nodes() / self.num_atoms))
        adjacency_list = [
            list(
                map(int, list(self.graph.neighbors(node)))
            )  # À améliorer sinon ce n'est pas des int mais des str
            for node in self.graph.nodes()
        ]
        n_cuts, membership = pymetis.part_graph(num_of_cuts, adjacency=adjacency_list)

        if print_progression:
            print("Partionned the graph")

        sub_graphs = []
        nodes_per_graph = []
        for i in range(num_of_cuts):
            nodes = np.argwhere(np.array(membership) == i).ravel()
            nodes_per_graph.append(nodes)
            sub_graphs.append(self.create_sub_graph(nodes))

        if print_progression:
            print("Sub_graphes created")

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

        if print_progression:
            print("MIS' done. Now combining")

        return self.combine_mis(MIS_list)
        # À vérifier ce bloc de for quand fonction de quantique sera à jour.

    def create_sub_graph(self, nodes: List[int]):
        # Trouver pourquoi la fonction subgraph marche pas
        subgraph = nx.Graph()
        subgraph.add_nodes_from(nodes)
        subgraph.add_edges_from(
            (u, v) for u, v in self.graph.edges(nodes) if u in nodes and v in nodes
        )
        return subgraph

    # CONCEPT DE RACINE??? PAS SUR, INSPIRÉ DE CHAT POUR DsITANCE LA PLUS LONGUE, noeud le plus éloigné
    def mis_tree(self, tree: nx.Graph):
        root = self.root_finder(tree)
        tree_directed = nx.bfs_tree(tree, root)
        with_node = np.ones(tree.number_of_nodes()) * np.nan
        without_node = np.ones(tree.number_of_nodes()) * np.nan
        root_indexes = [tree_directed.nodes()]
        mis_nodes = []

        def tree_mis_searcher(node):
            node_index = root_indexes.index(node)
            if nx.degree(tree_directed, node) == 1:
                with_node[node_index] = 1
                without_node[node_index] = 0
                return
            with_node[node_index] = 1
            without_node[node_index] = 0
            for i in tree_directed.successors(node):
                tree_mis_searcher(i)
                index_i = root_indexes.index(i)
                with_node[node_index] += without_node(index_i)
                without_node[node_index] += max(
                    without_node(index_i), with_node(index_i)
                )

        def mis_explorer(node, exclude_node):
            if nx.degree(tree_directed, node) == 1:
                if not exclude_node:
                    mis_nodes.append(node)
                return
            node_index = root_indexes.index(node)
            if exclude_node or without_node[node_index] > with_node[node_index]:
                for i in tree_directed.successors(node):
                    mis_explorer(i, exclude_node=False)
            else:
                mis_nodes.append(node)
                for i in tree_directed.successors(node):
                    mis_explorer(i, exclude_node=True)
            return

        tree_mis_searcher(root)
        mis_explorer(root, exclude_node=False)

        return mis_nodes

    def root_finder(self, tree):
        fartest_distances = np.empty(tree.number_of_nodes())
        start = tree.nodes()[0]
        for i, node in enumerate(tree.nodes):
            fartest_distances[i] = nx.shortest_path_length(tree, start, node)
        return np.argmax(fartest_distances)[0]

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

        new_mis = np.array([nx.isolate(forest)])
        forest.remove_edges_from(
            new_mis
        )  # Devrait marcher pour enlever nodes pas impliqués dans la combinaison

        for tree in nx.connected_components(forest):
            tree_graph = self.create_sub_graph(tree)
            result_mis = self.mis_tree(tree_graph)
            new_mis = np.append(new_mis, result_mis)

        return new_mis
