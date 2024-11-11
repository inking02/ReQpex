import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from numpy.typing import NDArray
from utils import create_node_dictionnary
import random


class Find_MIS_disks:
    def __init__(self, positions: NDArray[np.float_], radius: float) -> None:
        """
        points[:,0]=x;Longitudes
        points[:,1]=y;Latitudes
        """
        self.nodes_dict = create_node_dictionnary(positions)
        self.nb_nodes = np.shape(positions)[0]
        self.radius = radius
        self.create_graph()

    def create_graph(self):
        self.G = nx.Graph()
        for label, coord in self.nodes_dict.items():
            self.G.add_node(label, pos=coord)

        # Distance entre les positions
        def euclid_dist(pos1, pos2):
            return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

        # On met une condition pour ajouter des arêtes
        for i in self.G.nodes():
            for j in self.G.nodes():
                if (
                    i != j
                    and euclid_dist(self.G.nodes[i]["pos"], self.G.nodes[j]["pos"])
                    <= 2 * self.radius
                ):
                    self.G.add_edge(i, j)

    def run(
        self,
        shots: int,
        show_progress: bool = False,
        generate_histogram: bool = False,
        path: str = "",
        file_name: str = "MIS_histo.png",
    ):
        run_results = []
        zero_sting = "0" * self.nb_nodes
        seeds = [random.randint(0, 100000) for _ in range(shots)]
        for i in range(shots):
            I = nx.maximal_independent_set(self.G, seed=seeds[i])
            indexes = np.array(I, dtype=int)
            string_mis = zero_sting
            for j in indexes:
                string_mis = string_mis[:j] + "1" + string_mis[j + 1 :]
            run_results.append(string_mis)
            if show_progress:
                print("Shot ", i + 1, " done.")

        count_dict = {
            element: run_results.count(element) for element in set(run_results)
        }
        if generate_histogram:
            plt.hist(run_results)
            plt.savefig(path + file_name)
            plt.show()
        return count_dict
