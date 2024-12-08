from Quantum_MIS import Quantum_MIS
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
from pulse_utils import Pulse_constructor
import matplotlib.pyplot as plt
from pulser.devices import DigitalAnalogDevice, AnalogDevice

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
edges = np.array([(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (4, 5), (3, 4)])
G.add_edges_from(edges)


pos = nx.spring_layout(G, k = 0.1, seed=42)
nx.draw(G, with_labels = True, pos = pos)
plt.show()

Pulse = Pulse_constructor(4000, "Rise_fall")
device = AnalogDevice
test =  Quantum_MIS(G, device)
test.print_regs()

test.run(generate_histogram=True, Pulse = Pulse)

