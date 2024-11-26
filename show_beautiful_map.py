import numpy as np
import pandas as pd
import folium
from folium.plugins import MiniMap


def beaufiful_map_getter(path: str = "", show_estrie_aide=True, show: bool = False):
    new_bins_location = pd.read_csv(path + "datasets/nouvelles_cloches.csv", sep=",")
    new_bins_location_numpy = new_bins_location[["Longitude", "Latitude"]].to_numpy(
        dtype=float, copy=True
    )
    bins_og_used = pd.read_csv(path + "datasets/cloches_utiles.csv", sep=",")
    bins_og_used_numpy = bins_og_used[["Longitude", "Latitude"]].to_numpy(
        dtype=float, copy=True
    )

    og_bins = pd.read_csv(path + "datasets/cloches.csv", sep=";")

    estrie_aide = pd.read_csv(path + "datasets/estrieaide.csv", sep=",")

    og_bins_numpy = og_bins[["Longitude", "Latitude"]].to_numpy(dtype=float, copy=True)

    added_indexes = []

    for i, new_bin in enumerate(new_bins_location_numpy):
        matching_rows = og_bins_numpy[(og_bins_numpy == new_bin).all(axis=1)]
        if np.shape(matching_rows) is not tuple([1, 2]):
            added_indexes.append(i)

    removed_indexes = []

    for i, og_bin in enumerate(og_bins_numpy):
        matching_rows = bins_og_used_numpy[(bins_og_used_numpy == og_bin).all(axis=1)]
        print(matching_rows)
        if not np.shape(matching_rows)[0] == 1:
            removed_indexes.append(i)

    only_added_bins = new_bins_location.iloc[added_indexes]

    def show_map(show_estrie_aide: bool = False, show=False):
        sherbrooke_coord = [45.40198690041696, -71.88968408774863]
        my_map = folium.Map(location=sherbrooke_coord, zoom_start=13)
        minimap = MiniMap()
        my_map.add_child(minimap)
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
        for i, (_, row) in enumerate(og_bins.iterrows()):
            coords = [row["Longitude"], row["Latitude"]]
            coords[0], coords[1] = coords[1], coords[0]
            name = row["Nom de la borne"]
            adress = str(row["Addresse"]) + ", " + row["Rue"]
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
        my_map.save(path + "figures/map_with_stats.html")
        if show:
            my_map.show_in_browser()

    show_map(show_estrie_aide=show_estrie_aide, show=show)


path = "/Users/lf/Documents/GitHub/ReQpex/"
beaufiful_map_getter(path=path, show_estrie_aide=True, show=True)
