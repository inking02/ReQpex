"""
File that solve sthe recupex problem.
"""

import recupex_solver
from utils.generate_maps import recap_map_getter
from typing import List


def main(
    radius_simplify_bins: float,
    radius_simplify_locations: float,
    radius_new_distribution: float,
    save_maps: bool = False,
    show_maps: bool = False,
    bin_images: bool = False,
    save_maps_recap: bool = False,
    show_maps_recap: bool = False,
    show_estrie_aide: bool = False,
    use_quantum: bool = True,
    path: str = "",
    num_atoms: List[int] = [6, 4],
) -> None:
    """
    Runs the algorithm to solve Recupex bins placement optimisation problem.

    Parameters:
    - radius_simplify_bins (float): The radius in km to use to create the graph of the original bins distributions.
    - radius_simplify_locations (float): The radius in km to remove possible location that is closer that
                                         radius_simplify_locations km to a bin that stayed in the previous iteration.
    - radius_new_distribution: (float): The radius in km to use to create the graph of the simplified possible locations.
    - save_maps (bool = False): Whether or not to save the maps generated by the 3 steps of the optimisation.
    - show_maps (bool = False): Whether or not to show in the browser the maps generated by the 3 steps of the optimisation.
    - bin_images (bool = False): Whether or not to use recupex's bin image as the pins of the the 3 steps of the optimisation
    - save_maps_recap (bool = False): Whether or not to save the maps showing the result of the optimisation.
    - show_maps_recap (bool = False): Whether or not to show in the browser the maps showing the result of the optimisation.
    - show_estrie_aide (bool = False): Whether or not to show Estrie-Aide's bins on the recap map.
    - use_quantum (bool = True): Wheter or not to use the quantum implementation of the MIS solver. If not, the netwrokx's MIS function
                                 will be used.
    - path (str=""): The local path to the recupex directory (It includes the Recupex's folder).
    - num_atoms (List[int] = [6, 4]): The list of the maximum number of nodes in the subgraphes sent to the QMIS function, The element at the first
                                      position is for the bin MIS and the other one is for the new positions MIS.

    Returns:
    str: The key with the maximum value.
    """

    recupex_solver.simplify_bins(
        radius_simplify_bins,
        path=path,
        show_map=show_maps,
        save_map=save_maps,
        bin_image=bin_images,
        use_quantum=use_quantum,
        num_atoms=num_atoms[0],
    )
    print("Bins simplified")
    print("******************************************")

    recupex_solver.remove_possibles_new_locations(
        radius_simplify_locations,
        path=path,
        save_map=save_maps,
        show_map=show_maps,
        bin_image=bin_images,
    )
    print("Possible locations simplified")
    print("******************************************")

    recupex_solver.place_new_bins(
        radius_new_distribution,
        show_map=show_maps,
        save_map=save_maps,
        path=path,
        bin_image=bin_images,
        use_quantum=use_quantum,
        num_atoms=num_atoms[1],
    )
    print("New distribution calculated")
    print("******************************************")

    if save_maps_recap or show_maps_recap:
        recap_map_getter(
            path=path,
            show_estrie_aide=show_estrie_aide,
            show=show_maps_recap,
            save=save_maps_recap,
        )


if __name__ == "__main__":
    radius_simplify_bins = 1.5
    radius_simplify_locations = 1.5
    radius_new_distribution = 2.8

    path = "/Users/lf/Documents/GitHub/ReQpex/"

    save_maps = True
    show_maps = False
    bin_images = True
    save_maps_recap = False
    show_maps_recap = True
    show_estrie_aide = True
    use_quantum = True
    main(
        radius_simplify_bins=radius_simplify_bins,
        radius_simplify_locations=radius_simplify_locations,
        radius_new_distribution=radius_new_distribution,
        save_maps=save_maps,
        show_maps=show_maps,
        bin_images=bin_images,
        save_maps_recap=save_maps_recap,
        show_maps_recap=show_maps_recap,
        show_estrie_aide=show_estrie_aide,
        use_quantum=use_quantum,
        path=path,
    )
