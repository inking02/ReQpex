import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import networkx as nx
from array import *

# Lecture du fichier
points = pd.read_csv(
    "/Users/lf/Desktop/Université/Session 3/BSQ201/Projet 2/adresse_sherbrooke_ll.csv"
)

# Création d'un geoDataFrame pour manipulation plus simple pour une mise en graphique
gdf = gpd.GeoDataFrame(
    points, geometry=gpd.points_from_xy(points.Longitude, points.Latitude)
)

# On sort les coordonnées de chaque bac
points = gdf[["geometry"]].to_string(index=False)

# Utilisation d'une carte de Sherbrooke pour le visuel
sherbrooke = gpd.read_file(
    "/Users/lf/Desktop/Université/Session 3/BSQ201/Projet 2/Arrondissements/Arrondissements.shp"
)
map_sherbrooke = sherbrooke.to_crs(epsg=4326)

fig, ax = plt.subplots(figsize=(10, 10))

map_sherbrooke.plot(ax=ax, color="green", figsize=(10, 10))

gdf.plot(ax=ax, color="blue", label="Bacs", markersize=3)

ax.set_title("Bacs de récupération à Sherbrooke")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.legend()
plt.savefig("carte.png")
