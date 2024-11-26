import numpy as np
from Find_MIS_discs import Find_MIS_discs
from utils.generate_maps import generate_town_graph_connected
from utils.utils import disc_graph_to_connected
from Big_QMIS import BIG_QMIS
import matplotlib.pyplot as plt


def main(
    points,
    radius,
    shots,
    generate_graph=False,
    map_background=False,
    show_progress=False,
    generate_histogram=False,
    path="",
):
    if generate_graph:
        generate_town_graph_connected(
            points,
            radius=radius,
            title="Graph",
            path=path,
            map_background=map_background,
            file_name="figures/graph.png",
        )
    graph = disc_graph_to_connected(points, radius)
    MIS_solver = Find_MIS_discs(graph)
    res = MIS_solver.run(
        shots,
        show_progress=show_progress,
        generate_histogram=generate_histogram,
        path=path,
    )
    print(res)
    """Big"""
    plt.clf()
    big = BIG_QMIS(graph, num_atoms=2)
    big.run()


if __name__ == "__main__":
    points = np.empty((5, 2))
    points[:, 0] = [0.39808454, 0.05669836, -0.49947729, -0.95530561, 1.0]
    points[:, 1] = [0.92306714, -0.54467165, 0.22258244, -0.69346288, 0.09248494]

    radius = 1
    shots = 10
    generate_graph = True
    map_background = False
    show_progress = True
    generate_histogram = True
    path = "/Users/lf/Desktop/Université/Session 3/BSQ201/Projet 2/ReQpex/"
    main(
        points,
        radius,
        shots,
        generate_graph=generate_graph,
        map_background=map_background,
        show_progress=show_progress,
        generate_histogram=generate_histogram,
    )
