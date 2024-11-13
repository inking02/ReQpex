from scipy.spatial import distance_matrix
import numpy as np
import matplotlib.pyplot as plt


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

        return scaled_coords, scaled_radius

def find_unit_disk_radius(coords) -> float:
        """Calcul du rayon optimal pour que le graphe soit un graphe de disque unitaire."""
        distances = []
        for i, coord1 in enumerate(coords):
            for j, coord2 in enumerate(coords):
                if i < j and np.linalg.norm(coord1 - coord2) <= 1:  # Connexion possible dans un graphe unitaire
                    distances.append(np.linalg.norm(coord1 - coord2))
        
        # Rayon optimal pour que le graphe soit un graphe de disque unitaire
        unit_disk_radius = max(distances) if distances else 1.0  # 1.0 comme valeur par défaut
        return unit_disk_radius

def plot_histogram(count_dict, shots: int, file_name: str):
    most_freq = {k: v for k, v in count_dict.items() if v > 0.02*shots} 
    plt.bar(list(most_freq.keys()), list(most_freq.values()))
    plt.xticks(rotation="vertical")
    plt.ylabel('counts')
    plt.xlabel('bitstrings')
    plt.savefig(file_name)
    plt.show()