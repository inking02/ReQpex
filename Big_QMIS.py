"""
File containing the class of the method using the QMIS method on bigger graph to retrieve an independent set with a close to maximal number of nodes. 
The class of this MIS finder and its method are in this class.
"""

import numpy as np
import networkx as nx
from QMIS_code.QAA import QAA
from numpy.typing import NDArray
from typing import List, Callable
import pymetis
from QMIS_code.pulse_utils import pulse_constructor


class BIG_QMIS:
    def __init__(self, graph: nx.Graph, num_atoms: int = 10) -> None:
        """
        Object that can run the quantum analog computing BIG_QMIS algorithm. To create the object, networkx's graph architecture must be used.
        A number of atoms greater that 15 will not provide good results. It uses the QAA method to find a MIS on subgraphs. This method can be found
        in the QMIS_code.QAA.py file.

        Parameters:
        - self: BIG_QMIS MIS_Solver object to create.
        - graph (networkx.Graph): The graph to find an MIS on.
        - num_atoms (int=10): The maximum number of nodes (atoms in the pulser simulator) used in the subgraphs.

        Returns:
        - None
        """
        self.graph = graph
        if num_atoms > 20:
            self.num_atoms = 20
        else:
            self.num_atoms = num_atoms

    def max_bitstring(
        res_dict: dict, index_positions: NDArray[np.int_], other_info: List = []
    ) -> str:
        """
        Returns the key of a dictionary with the maximum value. The other arguments are not useful. They
        are just there as fillers to use as a general function in the class. In this class, the optimal key
        is the bitstring representing which nodes are used in the MIS.

        Parameters:
        - res_dict (dict): The counts dictionary of the result if the QAA algorithm.
        - index_positions (NDArray[np.int_]): The order of the nodes implied in the bitstring. It wont be used
                                              in this function, it is a filler argument.
        - other_info (List = []): Other info that may be useful to differentiate the answers to give the best key.
                                 It wont be used in this function, it is a filler argument.

        Returns:
        - str: The key with the maximum value.
        """
        return max(zip(res_dict.values(), res_dict.keys()))[1]

    def run(
        self,
        pulse: Callable = pulse_constructor(4000, "Rise_fall"),
        best_bitstring_getter: Callable = max_bitstring,
        shots: int = 1000,
        other_info: List = [],
        print_progression: bool = False,
        print_log_pulser: bool = False,
    ) -> List[str]:
        """
        Method to run the QAA algorithm on bigger graphs. A classical determinist algorithm will merge the sub graphs MIS' to a independent set
        with close to the maximum amount of nodes.

        Parameters:
        - self: BIG_QMIS MIS_Solver object to use.
        - Pulse (Callable = Pulse_constructor(4000, "Rise_fall")): A callable of a function returning a Pulse class object from Pulser's library. It is the pulse given to the set of
                                                                   the atoms to run the algorithm.
        - best_bitstring_getter (Callable = max_bitstring): The function that returns the best bitstring from the count dictionary given
                                                            by the QAA algorithm run function. It must take the result dictionary, the array that gives the order
                                                            of the nodes in the bitstrings (index_positions) and the other infos needed to differentiate the nodes (other_info).
        - shots (int = 100): The number of shots that each subgraph must be run on the pulser simulator.
        - other_info (List = []): The other information that must be used by the best_bitstring_getter function.
        - print_progression (bool = False): Whether or not to print the progression of the algorithm.
        - print_log_pulser (bool = False): Whether or not to print the log info of the runs on the pulser architecture.

        Returns:
        - List[str]: The list of the nodes implied in the MIS.
        """
        num_of_cuts = int(np.ceil(self.graph.number_of_nodes() / self.num_atoms))
        adjacency_list = [
            list(map(int, list(self.graph.neighbors(node))))
            for node in self.graph.nodes()
        ]
        membership = pymetis.part_graph(num_of_cuts, adjacency=adjacency_list)[1]

        if print_progression:
            print("Partitioned the graph")

        sub_graphs = []
        nodes_per_graph = []
        for i in range(num_of_cuts):
            nodes = np.argwhere(np.array(membership) == i).ravel()
            nodes = [str(j) for j in nodes]
            if not nodes == []:
                nodes_per_graph.append(nodes)
                sub_graphs.append(self.create_sub_graph(nodes))

        if print_progression:
            print("Subgraphs created")

        MIS_list = []

        for i, (graph, nodes) in enumerate(zip(sub_graphs, nodes_per_graph)):
            label_changer = dict()

            # Running the QAA algotithm on the subgraphs
            for k, node in enumerate(nodes):
                label_changer[node] = str(k)
            relabeled_graph = nx.relabel_nodes(graph, label_changer, copy=True)

            MIS_object = QAA(relabeled_graph)

            res_dict = MIS_object.run(pulse, shots=shots, progress_bar=print_log_pulser)
            best_bitstring = best_bitstring_getter(
                res_dict, nodes, other_info=other_info
            )
            # Can not have a MIS with no nodes implied
            if best_bitstring == "0" * len(nodes):
                del res_dict[best_bitstring]
                best_bitstring = best_bitstring_getter(
                    res_dict, nodes, other_info=other_info
                )
            # Mapping the result MIS to the correct node index
            independent_nodes = []
            for j in range(len(best_bitstring)):
                if best_bitstring[j] == "1":
                    independent_nodes.append(nodes[j])
            MIS_list.append(independent_nodes)

        if print_progression:
            print("MIS' done. Now combining")

        # Combining the sub-graphs' MIS into a total independent set
        result = self.combine_mis(MIS_list)
        return result

    def create_sub_graph(self, nodes: List[str]) -> nx.Graph:
        """
        - self: BIG_QMIS MIS_Solver object to use.
        Method to create a subgraph of the class' main graph with the networkx.Graph architecture.

        Parameters:
        - nodes (List[str]): The list of the nodes that need to be included in the subgraph

        Returns:
        - networkx.Graph: The sub graph create with the nodes specified.
        """
        subgraph = nx.Graph()
        subgraph.add_nodes_from(nodes)
        nodes_to_add = []
        for u, v in self.graph.edges():
            if u in nodes and v in nodes:
                nodes_to_add.append((u, v))
        subgraph.add_edges_from(nodes_to_add)

        return subgraph

    def mis_tree(self, tree: nx.Graph) -> List[str]:
        """
        Method that finds the MIS of a tree.

        Parameters:
        - self: BIG_QMIS MIS_Solver object to use.
        - tree (networkx.Graph): The tree that one of its MIS must be found.

        Returns:
        - List[str]: The list of the nodes implied in the MIS.
        """
        # Creating the directed tree with the root found
        root = self.root_finder(tree)
        tree_directed = nx.bfs_tree(tree, root)

        # Creating the  arrays to complete the dynamic algorithm
        with_node = np.empty(tree.number_of_nodes())
        without_node = np.empty(tree.number_of_nodes())

        indexes = [i for i in tree_directed.nodes()]
        mis_nodes = []

        def tree_mis_explorer(node: str):
            """
            Method that explores the tree to note the maximum number of nodes included in a MIS that includes or not the node
            given to the function.

            Parameters:
            - node (str): The node to determine the size of the MIS if its included or not.

            Returns:
            - None
            """
            node_index = indexes.index(node)
            if tree_directed.out_degree(node) == 0:
                with_node[node_index] = 1
                without_node[node_index] = 0
                return
            with_node[node_index] = 1
            without_node[node_index] = 0
            for i in tree_directed.successors(node):
                tree_mis_explorer(i)
                index_i = indexes.index(i)
                with_node[node_index] += without_node[index_i]
                without_node[node_index] += max(
                    without_node[index_i], with_node[index_i]
                )

        def tree_mis_searcher(node: str, exclude_node: bool) -> None:
            """
            With the dynamic arrays given earlier, The method find if the the node analyzed is the MIS or not.

            Parameters:
            - node (str): The node that which it is must be determined if it is included in the MIS.
            - exclude_node (bool): Whether or not the node is to be excluded in the MIS.

            Returns:
            - None
            """
            if tree_directed.out_degree(node) == 0:
                if not exclude_node:
                    mis_nodes.append(node)
                return
            node_index = indexes.index(node)
            if exclude_node or without_node[node_index] > with_node[node_index]:
                for i in tree_directed.successors(node):
                    tree_mis_searcher(i, exclude_node=False)
            else:
                mis_nodes.append(node)
                for i in tree_directed.successors(node):
                    tree_mis_searcher(i, exclude_node=True)
            return

        tree_mis_explorer(root)
        tree_mis_searcher(root, exclude_node=False)

        return mis_nodes

    def root_finder(self, tree: nx.Graph) -> str:
        """
        Method to find the root in a tree. SInce this notion is not really defined in an undirected graph,
        it is the node that is the "farthest" rom  the others.

        Parameters:
        - self: BIG_QMIS MIS_Solver object to use.
        - tree (networkx.Graph): The tree that its root must be found.

        Returns:
        - str: The node that is the root of the tree.
        """
        further_distances = []
        nodes_array = [i for i in tree.nodes()]
        start = nodes_array[0]
        for node in nodes_array:
            further_distances.append(nx.shortest_path_length(tree, start, node))
        root_index = np.argmax(further_distances)
        return nodes_array[root_index]

    def combine_mis(self, MIS_list: List[List[str]]) -> List[str]:
        """
        Method to combine the MIS' of all of the subgraphs into an independent set with the maximal amount of nodes given the sub-MIS'.

        Parameters:
        - self: BIG_QMIS MIS_Solver object to use.
        - MIS_list (List[List[str]]): The list if the MIS' of the subgraphs.

        Returns:
        - List[str]: The total MIS of the main graph.
        """
        # Separating the subgraphs into two halves to combined them afterwards when they each form one big subgraph
        if len(MIS_list) == 1:
            return MIS_list[0]
        n = len(MIS_list) // 2
        MIS_one = self.combine_mis(MIS_list[:n])
        MIS_two = self.combine_mis(MIS_list[n:])
        # Flattening the lists
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

        # Creating the forest generated between the vertices connected between the two subgraphs
        forest = nx.Graph()
        forest.add_nodes_from(MIS_one)
        forest.add_nodes_from(MIS_two)
        edges_to_add = [
            e
            for e in self.graph.edges
            if e[0] in MIS_one
            and e[1] in MIS_two
            or e[0] in MIS_two
            and e[1] in MIS_one
        ]

        forest.add_edges_from(edges_to_add)

        nodes_to_remove = [i for i in nx.isolates(forest)]
        if not nodes_to_remove == []:
            forest.remove_nodes_from(nodes_to_remove)

        # Finding the MIS of the forest
        if forest.number_of_edges() == 0:
            return nodes_to_remove
        new_mis = []
        for tree in nx.connected_components(forest):
            tree_graph = self.create_sub_graph(tree)
            result_mis = self.mis_tree(tree_graph)
            new_mis = np.append(new_mis, result_mis)
        return np.append(new_mis, nodes_to_remove)
