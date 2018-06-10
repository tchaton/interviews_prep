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

    def kruskal(self):
        '''
        Reference : https://en.wikipedia.org/wiki/Kruskal%27s_algorithm
        '''
        d = []
        h = {}
        seen = []
        for s in self.vertices.keys():
            for e in self.vertices[s].keys():
                n = s+e
                n1 = e+s
                if n not in seen:
                    seen.append(n)
                    seen.append(n1)
                    d.append([s+e, self.vertices[s][e]])
        sorted_vertexes = sorted(d, key=lambda x:x[1])[::-1]
        for v in sorted_vertexes:
            a, b = v[0]
            h[a] = 0
            h[b] = 0
        for index, key in enumerate(h.keys()):
            h[key] = index
        spanning_tree = []
        while len(sorted_vertexes) > 0:
            mimimum_conn = sorted_vertexes.pop()
            conn, weight = mimimum_conn
            s, e = conn
            a, b = h[s], h[e]
            if a !=b:
                for key in h:
                    if h[key] in [a, b]:
                        h[key] = a
                spanning_tree.append(conn)
        return sorted(spanning_tree)

class Test_Kruskal(unittest.TestCase):
    def test_Kruskal(self):
        g = Graph()
        g.add_vertex('A', {'B': 7, 'C': 8})
        g.add_vertex('B', {'A': 7, 'F': 2})
        g.add_vertex('C', {'A': 8, 'F': 6, 'G': 4})
        g.add_vertex('D', {'F': 8})
        g.add_vertex('E', {'H': 1})
        g.add_vertex('F', {'B': 2, 'C': 6, 'D': 8, 'G': 9, 'H': 3})
        g.add_vertex('G', {'C': 4, 'F': 9})
        g.add_vertex('H', {'E': 1, 'F': 3})
        print(g.kruskal())

if __name__ == '__main__':
    unittest.main()
