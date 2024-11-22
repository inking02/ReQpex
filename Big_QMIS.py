import numpy as np
import networkx as nx
from QMIS_code.Quantum_MIS import Quantum_MIS
from numpy.typing import NDArray
from typing import List
import pymetis


# À voir si on fait juste un graph en entree
class BIG_QMIS:
    def __init__(self, graph: nx.Graph, num_atoms: int = 10) -> None:
        self.graph = graph
        if num_atoms > 20:
            self.num_atoms = 20
        else:
            self.num_atoms = num_atoms

    def max_bitstring(res_dict, index_positions, other_info=[]):
        return max(zip(res_dict.values(), res_dict.keys()))[1]

    def run(
        self,
        pulse,
        best_bitstring_getter: callable = max_bitstring,
        other_info=[],
        print_progression: bool = False,
        print_log_pulser: bool = False,
    ):
        num_of_cuts = int(np.ceil(self.graph.number_of_nodes() / self.num_atoms))
        adjacency_list = [
            list(
                map(int, list(self.graph.neighbors(node)))
            )  # À améliorer sinon ce n'est pas des int mais des str
            for node in self.graph.nodes()
        ]
        membership = pymetis.part_graph(num_of_cuts, adjacency=adjacency_list)[1]

        if print_progression:
            print("Partionned the graph")

        sub_graphs = []
        nodes_per_graph = []
        for i in range(num_of_cuts):
            nodes = np.argwhere(np.array(membership) == i).ravel()
            nodes = [str(j) for j in nodes]
            if not nodes == []:
                nodes_per_graph.append(nodes)
                sub_graphs.append(self.create_sub_graph(nodes))

        if print_progression:
            print("Sub_graphes created")
            # print("The subgraphes:")
            # print(nodes_per_graph)

        MIS_list = []

        for i, (graph, nodes) in enumerate(zip(sub_graphs, nodes_per_graph)):
            nodes_index_to_value = [graph.nodes()]
            label_changer = dict()
            for k in nodes_index_to_value:
                label_changer["nodes_index_to_value[i]"] = k
            MIS_object = Quantum_MIS(
                nx.relabel_nodes(graph, label_changer, copy=True)
            )  # À faire
            res_dict = MIS_object.run(pulse, shots=100, progress_bar=print_log_pulser)
            best_bitstring = best_bitstring_getter(
                res_dict, nodes, other_info=other_info
            )
            if best_bitstring == "0" * len(nodes):
                del res_dict[best_bitstring]
                best_bitstring = best_bitstring_getter(
                    res_dict, nodes, other_info=other_info
                )

            independant_nodes = []
            for j in range(len(best_bitstring)):
                if best_bitstring[j] == "1":
                    independant_nodes.append(nodes[j])

            MIS_list.append(independant_nodes)
        if print_progression:
            print("MIS' done. Now combining")

        result = self.combine_mis(MIS_list)
        return result
        # À vérifier ce bloc de for quand fonction de quantique sera à jour.

    def create_sub_graph(self, nodes: List[int]):
        # Trouver pourquoi la fonction subgraph marche pas
        subgraph = nx.Graph()
        subgraph.add_nodes_from(nodes)
        subgraph.add_edges_from(
            tuple([u, v])
            for (u, v) in self.graph.edges(nodes)
            if u in nodes and v in nodes
        )
        return subgraph

    # CONCEPT DE RACINE??? PAS SUR, INSPIRÉ DE CHAT POUR DsITANCE LA PLUS LONGUE, noeud le plus éloigné
    def mis_tree(self, tree: nx.Graph):
        root = self.root_finder(tree)
        tree_directed = nx.bfs_tree(tree, root)
        with_node = np.empty(tree.number_of_nodes())
        without_node = np.empty(tree.number_of_nodes())
        indexes = [i for i in tree_directed.nodes()]
        mis_nodes = []

        def tree_mis_searcher(node):
            node_index = indexes.index(node)
            if tree_directed.out_degree(node) == 0:
                with_node[node_index] = 1
                without_node[node_index] = 0
                return
            with_node[node_index] = 1
            without_node[node_index] = 0
            for i in tree_directed.successors(node):
                tree_mis_searcher(i)
                index_i = indexes.index(i)
                with_node[node_index] += without_node[index_i]
                without_node[node_index] += max(
                    without_node[index_i], with_node[index_i]
                )

        def mis_explorer(node, exclude_node):
            if tree_directed.out_degree(node) == 0:
                if not exclude_node:
                    mis_nodes.append(node)
                return
            node_index = indexes.index(node)
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

    def root_finder(self, tree: nx.Graph):
        fartest_distances = []
        nodes_array = [i for i in tree.nodes()]
        start = nodes_array[0]
        for node in nodes_array:
            fartest_distances.append(nx.shortest_path_length(tree, start, node))
        root_index = np.argmax(fartest_distances)
        return nodes_array[root_index]

    def combine_mis(self, MIS_list: List[List[str]]):
        if len(MIS_list) == 1:
            return MIS_list[0]
        n = len(MIS_list) // 2
        MIS_one = self.combine_mis(MIS_list[:n])
        MIS_two = self.combine_mis(MIS_list[n:])
        MIS_one = (
            [node for sublist in MIS_one for node in sublist]
            if isinstance(MIS_one[0], list)
            else MIS_one
        )
        MIS_two = (
            [node for sublist in MIS_two for node in sublist]
            if isinstance(MIS_two[0], list)
            else MIS_two
        )

        # À vérifier mais devrait fonctionner
        forest = nx.Graph()
        forest.add_nodes_from(MIS_one)
        forest.add_nodes_from(MIS_two)
        forest.add_edges_from(
            (u, v)
            for u, v in self.graph.edges(forest.nodes())
            if u in MIS_one and v in MIS_two
        )

        nodes_to_remove = [i for i in nx.isolates(forest)]
        if not nodes_to_remove == []:
            forest.remove_nodes_from(
                nodes_to_remove
            )  # Devrait marcher pour enlever nodes pas impliqués dans la combinaison
        if forest.number_of_edges() == 0:
            return nodes_to_remove
        new_mis = []
        for tree in nx.connected_components(forest):
            tree_graph = self.create_sub_graph(tree)
            result_mis = self.mis_tree(tree_graph)
            new_mis = np.append(new_mis, result_mis)
        return np.append(new_mis, nodes_to_remove)
