# Implementatin of Dijkstra Algorithm

__author__ = 'tchaton'

import unittest
import sys
import heapq
import numpy as np
import networkx as nx

def has_key(d, key):
    try:
        a = d[key]
        return True
    except:
        return False

class Graph:

    def __init__(self):
        self.vertices = {}

    def add_vertex(self, name, edges):
        self.vertices[name] = edges

    def shortest_path(self, initial, finish):
        '''
        Reference : https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
        '''
        if not has_key(self.vertices, initial):
            raise Exception(str(initial)+ ' doesn t exist')
        if not has_key(self.vertices, finish):
            raise Exception(str(finish)+ ' doesn t exist')
        distances = {}
        previous = {}
        nodes = []

        for vertex in self.vertices:
            if vertex == initial:
                distances[vertex] = 0
                heapq.heappush(nodes, [0, vertex])
            else:
                distances[vertex] = np.inf
                heapq.heappush(nodes, [np.inf, vertex])
            previous[vertex] = None

        while nodes:
            closest = heapq.heappop(nodes)[1]
            if closest == finish:
                path = []
                while previous[closest]:
                    path.append(closest)
                    closest = previous[closest]
                return [initial] + path[::-1]
            if distances[closest] == np.inf:
                break

            for n_vertex in self.vertices[closest]:
                alt_dist = distances[closest] + self.vertices[closest][n_vertex]
                if alt_dist < distances[n_vertex]:
                    distances[n_vertex] = alt_dist
                    previous[n_vertex] = closest
                    for n in nodes:
                        if n[1] == n_vertex:
                            n[0] = alt_dist
                            break
                    heapq.heapify(nodes)
        return distances

class Test_Dijkstra(unittest.TestCase):

    def test_shortest_path(self):
        g = Graph()
        g.add_vertex('A', {'B': 7, 'C': 8})
        g.add_vertex('B', {'A': 7, 'F': 2})
        g.add_vertex('C', {'A': 8, 'F': 6, 'G': 4})
        g.add_vertex('D', {'F': 8})
        g.add_vertex('E', {'H': 1})
        g.add_vertex('F', {'B': 2, 'C': 6, 'D': 8, 'G': 9, 'H': 3})
        g.add_vertex('G', {'C': 4, 'F': 9})
        g.add_vertex('H', {'E': 1, 'F': 3})
        G = nx.Graph(g.vertices)
        self.assertEqual(g.shortest_path('A', 'H'), nx.dijkstra_path(G, 'A', 'H'))

if __name__ == '__main__':
    unittest.main()
