from scipy.spatial import distance_matrix
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def scale_coordinates(radius, coordinates, min_distance, max_distance):
        # Calcul des distances entre les points pour estimer l'échelle
        dist_matrix = distance_matrix(coordinates, coordinates)
        np.fill_diagonal(dist_matrix, np.inf)  # Ignore la diagonale
        min_dist = dist_matrix.min()  # Distance minimale dans les données d'origine

        # Calcul du facteur d'échelle pour assurer que la plus petite distance soit au moins `min_distance`
        scale_factor = min_distance / min_dist
        
        # Applique le facteur d'échelle
        scaled_coords = coordinates * scale_factor
        scaled_radius = radius*scale_factor

        # Recentre les coordonnées pour qu'elles restent proches de l'origine si nécessaire
        center_x = np.mean(scaled_coords[:, 0])
        center_y = np.mean(scaled_coords[:, 1])
        scaled_coords -= np.array([center_x, center_y])

        # Ajuste à nouveau si certains points dépassent la limite de `max_distance`
        max_dist_from_center = np.max(np.linalg.norm(scaled_coords, axis=1))
        if max_dist_from_center > max_distance:
            scale_factor = max_distance / max_dist_from_center
            scaled_coords *= scale_factor
            scaled_radius *=scale_factor

        return scaled_coords, scaled_radius


def find_minimal_radius(G: nx.Graph, pos):
    """
    Trouve la distance minimale entre deux sommets reliés par une arête dans un graphe.
    
    :param G: Un graphe NetworkX.
    :param coords: Un dictionnaire contenant les coordonnées des sommets {node: (x, y)}.
    :return: La distance minimale entre deux sommets reliés par une arête.
    """
    max_distance = 0  # Initialise la distance minimale avec une valeur infinie
    
    for u, v in G.edges():  # Parcourt toutes les arêtes du graphe
        # Récupère les coordonnées des deux sommets
        coord_u = np.array(pos[u])
        coord_v = np.array(pos[v])
        
        # Calcule la distance euclidienne entre les deux sommets
        distance = euclid_dist(coord_u, coord_v)
        
        # Met à jour la distance minimale
        if distance > max_distance:
            max_distance = distance
    
    return max_distance
     

def plot_histogram(count_dict, shots: int, file_name: str):
    most_freq = {k: v for k, v in count_dict.items() if v > 0.02*shots} 
    plt.bar(list(most_freq.keys()), list(most_freq.values()))
    plt.xticks(rotation="vertical")
    plt.ylabel('counts')
    plt.xlabel('bitstrings')
    plt.savefig(file_name)
    plt.show()

def euclid_dist(pos1, pos2):
    return ((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)**0.5