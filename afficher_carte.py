import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import networkx as nx
from array import *
from matplotlib.patches import Circle


def create_dictionnary(points):
    res = dict()
    longitudes = points.Longitude
    latitudes = points.Latitude
    for i in range(np.shape(points)[0]):
        res[str(i)] = (longitudes[i], latitudes[i])
    return res


def generate_map(
    points,
    title: str = "",
    path="",
    figsize=(10, 10),
    markersize=10,
    file_name="datasets/carte.png",
):
    # Création d'un geoDataFrame pour manipulation plus simple pour une mise en graphique
    gdf = gpd.GeoDataFrame(
        points, geometry=gpd.points_from_xy(points.Longitude, points.Latitude)
    )

    # On sort les coordonnées de chaque bac
    points = gdf[["geometry"]].to_string(index=False)

    # Utilisation d'une carte de Sherbrooke pour le visuel
    sherbrooke = gpd.read_file(path + "datasets/Arrondissements/Arrondissements.shp")
    map_sherbrooke = sherbrooke.to_crs(epsg=4326)

    fig, ax = plt.subplots(figsize=figsize)

    map_sherbrooke.plot(ax=ax, color="grey", figsize=figsize)

    gdf.plot(ax=ax, color="red", label="Bacs", markersize=markersize)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.legend()
    plt.savefig(path + file_name)


def generate_town_graph_radius(
    points,
    radius: float = 0.01,
    title: str = "",
    map_background=False,
    figsize=(10, 10),
    path="",
    file_name="datasets/graphe_cercles.png",
):
    pos = create_dictionnary(points)
    data = {"x": [], "y": [], "label": []}
    for label, coord in pos.items():
        data["x"].append(coord[0])
        data["y"].append(coord[1])
        data["label"].append(label)

    if map_background:
        sherbrooke = gpd.read_file(
            path + "datasets/Arrondissements/Arrondissements.shp"
        )
        map_sherbrooke = sherbrooke.to_crs(epsg=4326)

        fig, ax = plt.subplots(figsize=figsize)

        map_sherbrooke.plot(ax=ax, color="grey", figsize=figsize)

    for label, coord in pos.items():
        circle = Circle(
            (coord[0], coord[1]), radius=radius, edgecolor="red", facecolor="none"
        )
        plt.gca().add_patch(circle)

    # add labels
    for label, x, y in zip(data["label"], data["x"], data["y"]):
        plt.annotate(label, xy=(x, y))

    plt.scatter(data["x"], data["y"], marker="o")
    plt.title(label=title)
    plt.savefig(path + file_name)


def generate_town_graph_connected(
    points,
    radius: float = 0.01,
    title: str = "",
    map_background=False,
    figsize=(10, 10),
    path="",
    file_name="datasets/graphe_connecte.png",
):
    pos = create_dictionnary(points)
    data = {"x": [], "y": [], "label": []}
    for label, coord in pos.items():
        data["x"].append(coord[0])
        data["y"].append(coord[1])
        data["label"].append(label)

    # Graphe associé aux positions
    G = nx.Graph()

    for label, coord in pos.items():
        G.add_node(label, pos=coord)

    # Distance entre les positions
    def euclid_dist(pos1, pos2):
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    # On met une condition pour ajouter des arêtes
    for i in G.nodes():
        for j in G.nodes():
            if (
                i != j
                and euclid_dist(G.nodes[i]["pos"], G.nodes[j]["pos"]) <= 2 * radius
            ):  # Vu qu'on a des disques rights??? Dès que les deux se touchent ça compte??
                G.add_edge(i, j)

    # Plot
    if map_background:
        sherbrooke = gpd.read_file(
            path + "datasets/Arrondissements/Arrondissements.shp"
        )
        map_sherbrooke = sherbrooke.to_crs(epsg=4326)

        fig, ax = plt.subplots(figsize=figsize)

        map_sherbrooke.plot(ax=ax, color="grey", figsize=figsize)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    nx.draw(G, pos, with_labels=True, node_size=100)
    plt.savefig(path + file_name)
    # Pourquoi 2 graphiques


# Lecture du fichier
path = "/Users/lf/Desktop/Université/Session 3/BSQ201/Projet 2/ReQpex/"
"""
points = pd.read_csv(
    path+"datasets/adresse_sherbrooke_ll.csv"
)
"""

points = pd.read_csv(path + "datasets/liste_occupants_simple.csv")
r = 0.01
map_back = True

generate_map(points, title="Adresses", file_name="datasets/commerces.png")
plt.clf()
generate_town_graph_radius(
    points,
    title="Graphe des villes",
    map_background=map_back,
    path=path,
    radius=r,
    file_name="datasets/commerces_cercles.png",
)

plt.clf()
generate_town_graph_connected(
    points,
    title="Graphe des cloches",
    map_background=map_back,
    path=path,
    radius=r,
    file_name="datasets/commerces_connect.png",
)
