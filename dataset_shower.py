import pandas as pd
import matplotlib.pyplot as plt
import generate_maps


# Lecture du fichier
path = "/Users/lf/Desktop/Université/Session 3/BSQ201/Projet 2/ReQpex/"

points = pd.read_csv(path + "datasets/adresse_sherbrooke_ll.csv")

points = pd.read_csv(path + "datasets/liste_occupants_simple.csv")
points = points[["Longitude", "Latitude"]].to_numpy(dtype=float, copy=True)
r = 0.01
map_back = True


# Generate the maps
generate_maps.generate_map(points, title="Adresses", file_name="maps/commerces.png")
plt.clf()
generate_maps.generate_town_graph_radius(
    points,
    title="Graphe des villes",
    map_background=map_back,
    path=path,
    radius=r,
    file_name="maps/commerces_cercles.png",
)

plt.clf()
generate_maps.generate_town_graph_connected(
    points,
    title="Graphe des cloches",
    map_background=map_back,
    path=path,
    radius=r,
    file_name="maps/commerces_connect.png",
)
