import numpy as np
from Find_MIS_disks import Find_MIS_disks
from generate_maps import generate_town_graph_connected


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
            title="Test",
            path=path,
            map_background=map_background,
            file_name="graph.png",
        )
    MIS_solver = Find_MIS_disks(points, radius)
    res = MIS_solver.run(
        shots,
        show_progress=show_progress,
        generate_histogram=generate_histogram,
        path=path,
    )
    print(res)


if __name__ == "__main__":
    points = np.empty((5, 2))
    points[:, 0] = [0.39808454, 0.05669836, -0.49947729, -0.95530561, 1.0]
    points[:, 1] = [0.92306714, -0.54467165, 0.22258244, -0.69346288, 0.09248494]

    radius = 1
    shots = 10
    generate_graph = False
    map_background = False
    show_progress = True
    generate_histogram = True
    path = "/Users/lf/Desktop/Université/Session 3/BSQ201/Projet 2/ReQpex/maps/"
    main(
        points,
        radius,
        shots,
        generate_graph=generate_graph,
        map_background=map_background,
        show_progress=show_progress,
        generate_histogram=generate_histogram,
    )
