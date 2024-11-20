from Quantum_MIS import Quantum_MIS
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
from QMIS_utils import Pulse_constructor

def random_UD_graph(nqubits, seed):
    np.random.seed(seed)
    r_absolute_min = 1
    blockade_radius = 6.4/5
    coords = [np.random.uniform(0, 1, size=(2))]
    while len(coords) < nqubits:
        cond = True
        while cond:
            #choose a random site to connect to. 
            ind = np.random.randint(0,len(coords))
            initial_pos = coords[ind]
            #propose new site:
            shift_vector_r = np.random.uniform(r_absolute_min, blockade_radius)
            angle = 2*np.pi*np.random.rand()
            new_pos = initial_pos + np.array([shift_vector_r*np.cos(angle), shift_vector_r*np.sin(angle)])
            #check min
            R_min = np.min(pdist(coords + [new_pos]))
            if R_min > r_absolute_min:
                #accept new coordinate
                cond = False
        
        coords.append(new_pos)

    graph = nx.Graph()
    edges = KDTree(coords).query_pairs(blockade_radius)
    graph.add_edges_from(edges)

    return graph

G = nx.Graph()
edges = np.array([(1, 2), (1, 3), (2,3), (3, 4), (3, 5),(4, 5), (5, 6)])
G.add_edges_from(edges)

Pulse = Pulse_constructor(4000, "Waveform")

test =  Quantum_MIS(G)
test.print_reg()

test.run(generate_histogram=True, Pulse = Pulse)

