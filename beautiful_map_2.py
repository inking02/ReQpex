import numpy as np
import pandas as pd
import folium
from folium.plugins import MiniMap

path = "/Users/lf/Documents/GitHub/ReQpex/"
show = False
save = True

new_distribution_quantum = pd.read_csv(
    path + "datasets/nouvelles_cloches_quantum.csv", sep=","
)
new_distribution_quantum_numpy = new_distribution_quantum[
    ["Longitude", "Latitude"]
].to_numpy(dtype=float, copy=True)

new_distribution_classical = pd.read_csv(
    path + "datasets/nouvelles_cloches_classical.csv", sep=","
)
new_distribution_classical_numpy = new_distribution_classical[
    ["Longitude", "Latitude"]
].to_numpy(dtype=float, copy=True)


same_bin_q = []
for i, q_bin in enumerate(new_distribution_quantum_numpy):
    matching_rows = new_distribution_classical_numpy[
        (new_distribution_classical_numpy == q_bin).all(axis=1)
    ]
    if np.shape(matching_rows)[0] == 1:
        same_bin_q.append(i)

only_same_bins = new_distribution_quantum.iloc[same_bin_q]


same_bin_c = []
for i, c_bin in enumerate(new_distribution_classical_numpy):
    matching_rows = new_distribution_quantum_numpy[
        (new_distribution_quantum_numpy == c_bin).all(axis=1)
    ]
    if np.shape(matching_rows)[0] == 1:
        same_bin_c.append(i)


def show_map(show: bool = False, save: bool = False):
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

    # Adding the bins that were the same
    for _, row in only_same_bins.iterrows():
        coords = [row["Longitude"], row["Latitude"]]
        coords[0], coords[1] = coords[1], coords[0]
        name = row["Nom de la borne"]
        adress = str(row["Addresse"]) + ", " + row["Rue"]
        html = f"""
        <h1> {name}</h1>
        <p>Adresse : {adress}</p>
        <p>Cette cloche est dans les deux distributions</p>
        """
        popup = folium.Popup(html=html, max_width=1000)
        folium.Marker(coords, popup=popup, icon=folium.Icon(color="blue")).add_to(
            my_map
        )

    # Adding the bins only in quantum
    for i, (_, row) in enumerate(new_distribution_quantum.iterrows()):
        # Checking if the bin was removed of not
        if not i in same_bin_q:
            coords = [row["Longitude"], row["Latitude"]]
            coords[0], coords[1] = coords[1], coords[0]
            name = row["Nom de la borne"]
            adress = str(row["Addresse"]) + ", " + row["Rue"]
            html = f"""
            <h1> {name}</h1>
            <p>Adresse : {adress}</p>
            <p>Cette cloche apparait juste en quantique</p>
            """
            popup = folium.Popup(html=html, max_width=1000)
            folium.Marker(coords, popup=popup, icon=folium.Icon(color="green")).add_to(
                my_map
            )
    # Adding the bins only in quantum
    for i, (_, row) in enumerate(new_distribution_classical.iterrows()):
        # Checking if the bin was removed of not
        if not i in same_bin_c:
            coords = [row["Longitude"], row["Latitude"]]
            coords[0], coords[1] = coords[1], coords[0]
            name = row["Nom de la borne"]
            adress = str(row["Addresse"]) + ", " + row["Rue"]
            html = f"""
            <h1> {name}</h1>
            <p>Adresse : {adress}</p>
            <p>Cette cloche apparait juste en classique</p>
            """
            popup = folium.Popup(html=html, max_width=1000)
            folium.Marker(coords, popup=popup, icon=folium.Icon(color="red")).add_to(
                my_map
            )
    print("Color code")
    print("A blue pin is a bin that is in both distribution")
    print("A green pin is a bin that is only in quantum")
    print("A red pin is a bin that is only in classical")

    print()
    # Save and or show the map
    if save:
        my_map.save(path + "figures/map_with_both_distributions.html")
    if show:
        my_map.show_in_browser()


show_map(show=show, save=save)
