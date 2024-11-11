import numpy as np


def create_node_dictionnary(points):
    res = dict()
    longitudes = points[:, 0]
    latitudes = points[:, 1]
    for i in range(np.shape(points)[0]):
        res[str(i)] = (longitudes[i], latitudes[i])
    return res
