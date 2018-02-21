import networkx
import matplotlib.pyplot as plt
from itertools import combinations
from random import randint

vector = {}
persons = ['a', 'b', 'c', 'd', 'e']

for person in persons:
    vector[person] = []

for man_pair in combinations(persons, 2):
    man1, man2 = man_pair
    r = randint(1, 10)
    if r % 2:
        continue
    else:
        vector[man1].append(man2)
        edge_labels[(man1, man2)] = r

graph = networkx.Graph(vector)
pos = networkx.spring_layout(graph)


networkx.draw_networkx_nodes(graph, pos)
networkx.draw_networkx_edges(graph, pos)
networkx.draw_networkx_labels(graph, pos)

plt.show()
