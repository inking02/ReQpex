import recupex_solver
from recap_map import recap_map_getter


def main(
    save_maps: bool = False,
    show_maps: bool = False,
    bin_images: bool = False,
    save_maps_recap: bool = False,
    show_maps_recap: bool = False,
    show_estrie_aide: bool = False,
    use_quantum: bool = True,
    path="",
):

    recupex_solver.simplify_bins(
        1.5,
        path=path,
        show_map=show_maps,
        save_map=save_maps,
        bin_image=bin_images,
        use_quantum=use_quantum,
    )
    print("Bins simplified")
    print("******************************************")

    recupex_solver.remove_possibles_new_locations(
        1.5, path=path, save_maps=save_maps, show_map=show_maps, bin_image=bin_images
    )
    print("Possible locations simplified")
    print("******************************************")

    recupex_solver.place_new_bins(
        2.8,
        show_map=show_maps,
        save_map=save_maps,
        path=path,
        bin_image=bin_images,
        use_quantum=use_quantum,
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
    path = "/Users/lf/Documents/GitHub/ReQpex/"
    save_maps = True
    show_maps = False
    bin_images = True
    save_maps_recap = True
    show_maps_recap = False
    show_estrie_aide = True
    use_quantum = True
    main(
        save_maps=save_maps,
        show_maps=show_maps,
        bin_images=bin_images,
        save_maps_recap=save_maps_recap,
        show_maps_reca=show_maps_recap,
        show_estrie_aide=show_estrie_aide,
        use_quantum=use_quantum,
        path=path,
    )
