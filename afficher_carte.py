import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import networkx as nx
from array import *
from matplotlib.patches import Circle


def generate_map(points, title: str = "", path="", figsize=(10, 10), markersize=10):
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
    plt.savefig(path + "datasets/carte.png")


def generate_town_graph(
    points,
    radius: float = 0.0000000001,
    title: str = "",
    map_background=False,
    figsize=(10, 10),
    path="",
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
    plt.show()
    # Pourquoi 2 graphiques


def create_dictionnary(points):
    res = dict()
    longitudes = points.Longitude
    latitudes = points.Latitude
    for i in range(np.shape(points)[0]):
        res[str(i)] = (longitudes[i], latitudes[i])
    return res


# Lecture du fichier
path = "/Users/lf/Desktop/Université/Session 3/BSQ201/Projet 2/ReQpex/"
"""
points = pd.read_csv(
    path+"datasets/adresse_sherbrooke_ll.csv"
)
"""

points = pd.read_csv(path + "/datasets/cloches.csv")

# generate_map(points, title="Adresses")
generate_town_graph(
    points, radius=1.0, title="Graphe des cloches", map_background=True, path=path
)
