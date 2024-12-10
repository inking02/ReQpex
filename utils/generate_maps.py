
"""
File containing the various functions to show the locations on a sherbrooke map.
"""

import numpy as np
import pandas as pd
import folium
from folium.plugins import MiniMap


def interactive_map(
    data_frame_to_show: pd.DataFrame,
    bin_image: bool = False,
    path: str = "",
    show_map: bool = False,
    save_map: bool = False,
    file_name: str = "map",
):
    """
    Creates a centered on Sherbrooke city with the given data to show.

    Parameters:
    - data_frame_to_show (pd.DataFrame): The pandas dataframe to use to create the map. It must have a Longitude,
      Latitude, Nom de la borne, Addresse and Rue columns.
    - bin_image (bool = False): Whether to show the pins as Recupex's bins or not.
    - path (str)=""): The local file to the REQPEX directory.
    - show_map (bool = False): Whether to show the map on the browser or not.
    - save_map (bool = False): Whether to save the map on the datasets' folder or not.
    - file_name (str = "map"): The name that the map must have. It must not include the extension.

    Returns:
    None
    """
    sherbrooke_coord = [45.40198690041696, -71.88968408774863]
    my_map = folium.Map(location=sherbrooke_coord, zoom_start=13)
    minimap = MiniMap()
    my_map.add_child(minimap)
    for _, row in data_frame_to_show.iterrows():
        coords = [row["Longitude"], row["Latitude"]]
        coords[0], coords[1] = coords[1], coords[0]
        name = row["Nom de la borne"]
        adress = str(row["Addresse"]) + ", " + row["Rue"]
        html = f"""
        <h1> {name}</h1>
        <p>Adresse : {adress}</p>
        """
        popup = folium.Popup(html=html, max_width=1000)
        if bin_image:
            icon = folium.features.CustomIcon(
                icon_image=path + "datasets/image_cloche_recupex.png",
                icon_size=(30, 30),
            )
            folium.Marker(coords, popup=popup, icon=icon).add_to(my_map)
        else:
            folium.Marker(coords, popup=popup).add_to(my_map)
    if save_map:
        my_map.save(path + "figures/" + file_name + ".html")
    if show_map:
        my_map.show_in_browser()


def recap_map_getter(
    path: str = "",
    show_estrie_aide: bool = True,
    show: bool = False,
    save: bool = False,
) -> None:
    """
    Method to create a map showing the bins that stayed, were removed and added. If wanted, Estrie-Aide's
    bins can be added to the map as well.

    Parameters:
    - path (str = ""): The local path to the recupex directory (It includes the Recupex's folder).
    - show_estrie_aide (bool = True): Whether or not to show Estrie-Aide's bins on the map.
    - show (bool = False): Whether of not to show the map on the browser.
    - save (bool = False): Whether of not to save the map on the "map_with_stats.html" file in the figures directory.

    Returns:
    None
    """
    # Loading the data
    new_bins_location = pd.read_csv(path + "datasets/new_bins.csv", sep=",")
    new_bins_location_numpy = new_bins_location[["Longitude", "Latitude"]].to_numpy(
        dtype=float, copy=True
    )
    bins_og_used = pd.read_csv(path + "datasets/bins_utils.csv", sep=",")
    bins_og_used_numpy = bins_og_used[["Longitude", "Latitude"]].to_numpy(
        dtype=float, copy=True
    )

    og_bins = pd.read_csv(path + "datasets/cloches.csv", sep=";")

    estrie_aide = pd.read_csv(path + "datasets/estrieaide.csv", sep=",")

    og_bins_numpy = og_bins[["Longitude", "Latitude"]].to_numpy(dtype=float, copy=True)

    # Finding bins that were added to the distribution
    added_indexes = []

    for i, new_bin in enumerate(new_bins_location_numpy):
        matching_rows = og_bins_numpy[(og_bins_numpy == new_bin).all(axis=1)]
        if np.shape(matching_rows) is not tuple([1, 2]):
            added_indexes.append(i)

    removed_indexes = []

    # Finding bins that were removed to the distribution
    for i, og_bin in enumerate(og_bins_numpy):
        matching_rows = bins_og_used_numpy[(bins_og_used_numpy == og_bin).all(axis=1)]
        if not np.shape(matching_rows)[0] == 1:
            removed_indexes.append(i)

    only_added_bins = new_bins_location.iloc[added_indexes]

    # Function to create the map
    def show_map(
        show_estrie_aide: bool = False, show: bool = False, save: bool = False
    ):
        sherbrooke_coord = [45.40198690041696, -71.88968408774863]
        my_map = folium.Map(location=sherbrooke_coord, zoom_start=13)
        minimap = MiniMap()
        my_map.add_child(minimap)
        """
        Method that creates the map object.

        Parameters:
        - show_estrie_aide (bool = True): Whether or not to show Estrie-Aide's bins on the map.
        - show (bool = False): Whether of not to show the map on the browser.
        - save (bool = False): Whether of not to save the map on the "map_with_stats.html" file in the figures directory.

        Returns:
        None
        """

        # Adding the bins that were added to the map
        for _, row in only_added_bins.iterrows():
            coords = [row["Longitude"], row["Latitude"]]
            coords[0], coords[1] = coords[1], coords[0]
            name = row["Nom de la borne"]
            adress = str(row["Addresse"]) + ", " + row["Rue"]
            html = f"""
            <h1> {name}</h1>
            <p>Adresse : {adress}</p>
            <p>Cette cloche sera ajoutée</p>
            """
            popup = folium.Popup(html=html, max_width=1000)
            folium.Marker(coords, popup=popup, icon=folium.Icon(color="green")).add_to(
                my_map
            )

        # Adding the bins that stayed or were removed
        for i, (_, row) in enumerate(og_bins.iterrows()):
            coords = [row["Longitude"], row["Latitude"]]
            coords[0], coords[1] = coords[1], coords[0]
            name = row["Nom de la borne"]
            adress = str(row["Addresse"]) + ", " + row["Rue"]

            # Checking if the bin was removed of not
            if i in removed_indexes:
                html = f"""
                <h1> {name}</h1>
                <p>Adresse : {adress}</p>
                <p>Cette cloche sera retirée</p>
                """
                popup = folium.Popup(html=html, max_width=1000)
                folium.Marker(
                    coords, popup=popup, icon=folium.Icon(color="red")
                ).add_to(my_map)
            else:
                html = f"""
                <h1> {name}</h1>
                <p>Adresse : {adress}</p>
                <p>Cette cloche restera ici</p>
                """
                popup = folium.Popup(html=html, max_width=1000)
                folium.Marker(
                    coords, popup=popup, icon=folium.Icon(color="blue")
                ).add_to(my_map)
        print("Color code")
        print("A blue pin is a bin that will stay")
        print("A green pin is a bin that will be added")
        print("A red pin is a bin that will be removed")

        # Adding Estrie-Aide's bins if necessary
        if show_estrie_aide:
            for _, row in estrie_aide.iterrows():
                coords = [row["Longitude"], row["Latitude"]]
                coords[0], coords[1] = coords[1], coords[0]
                name = row["Nom de la borne"]
                adress = str(row["Addresse"]) + ", " + row["Rue"]
                html = f"""
                <h1> {name}</h1>
                <p>Adresse : {adress}</p>
                <p>Cette cloche est une d'Estrie-Aide</p>
                """
                popup = folium.Popup(html=html, max_width=1000)
                folium.Marker(
                    coords, popup=popup, icon=folium.Icon(color="purple")
                ).add_to(my_map)

            print("A purple pin is an Estrie-Aide bin")

        print()
        # Save and or show the map
        if save:
            my_map.save(path + "figures/map_with_stats.html")
        if show:
            my_map.show_in_browser()

    show_map(show_estrie_aide=show_estrie_aide, show=show, save=save)
