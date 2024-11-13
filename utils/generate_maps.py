import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import networkx as nx
from array import *
from matplotlib.patches import Circle
from utils.utils import disc_graph_to_connected, create_node_dictionnary
from numpy.typing import NDArray


def generate_map(
    points: NDArray[np.float_],
    title: str = "",
    path="",
    figsize=(10, 10),
    markersize=10,
    file_name="figures/map.png",
):
    # Création d'un geoDataFrame pour manipulation plus simple pour une mise en graphique
    gdf = gpd.GeoDataFrame(
        points, geometry=gpd.points_from_xy(points[:, 0], points[:, 1])
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
    points: NDArray[np.float_],
    radius: float = 0.01,
    title: str = "",
    map_background=False,
    figsize=(10, 10),
    path="",
    file_name="figures/circle_graph.png",
):
    pos = create_node_dictionnary(points)
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
    points: NDArray[np.float_],
    radius: float = 0.01,
    title: str = "",
    map_background=False,
    figsize=(10, 10),
    path="",
    file_name="figures/connected_graph.png",
):
    pos = create_node_dictionnary(points)
    data = {"x": [], "y": [], "label": []}
    for label, coord in pos.items():
        data["x"].append(coord[0])
        data["y"].append(coord[1])
        data["label"].append(label)

    # Graphe associé aux positions
    G = disc_graph_to_connected(points, radius)

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
    nx.draw(G, pos=pos, with_labels=True, node_size=100)
    plt.savefig(path + file_name)
