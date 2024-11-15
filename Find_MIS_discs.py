import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from numpy.typing import NDArray
from utils.utils import disc_graph_to_connected
import random


class Find_MIS_discs:
    def __init__(self, graph: nx.Graph) -> None:
        """
        points[:,0]=x;Longitudes
        points[:,1]=y;Latitudes
        """
        self.G = graph

    def run(
        self,
        shots: int,
        show_progress: bool = False,
        generate_histogram: bool = False,
        path: str = "",
        file_name: str = "figures/MIS_histo.png",
    ):
        run_results = []
        zero_sting = "0" * self.G.number_of_nodes()
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
            plt.clf()
            plt.hist(run_results)
            plt.savefig(path + file_name)
        return count_dict
