"""
File containing the class of the classical MIS finder method. The class of this classifier and its method are in this class.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random


class Find_MIS_discs:
    def __init__(self, graph: nx.Graph) -> None:
        """
        Object that can run the classical MIS algorithm. To create the object, networkx's graph architecture must be used.

        Parameters:
        - graph (networkx.Graph): The graph to find an MIS on.

        Returns:
        None
        """
        self.G = graph

    def run(
        self,
        shots: int,
        show_progress: bool = False,
        generate_histogram: bool = False,
        path: str = "",
        file_name: str = "figures/MIS_histo.png",
    ) -> dict:
        """
        Method to run the classical MIS algorithm. For the number of shots, it will use a different seed of networkx's maximal_independent_set function.

        Parameters:
        - shots (int): The number of times the algotihm must be runned.
        - show_progress (bool = False): Whether or not to print the progress of the iterations of the algorithms.
        - generate_histogram (bool = False): Generate the result histogram of the runs of the algorithms.
        - path (str)=""): The local file to the REQPEX directory.
        - file_name (str = "figures/MIS_histo.png"): The file name that the histogram must be saved as. The filename must also include its local path from the
                                              ReQpex directory and use the extension png.

        Returns:
        dict: The counts dictionnary of the results from the shots of the algorithms.
        """
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
