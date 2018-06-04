# Find a route from the first node to the second node in a directed graph.

__author__ = 'tchaton'
import numpy as np

def find_route(node1, node2):
    seen = []
    real_path = []
    def _find_route(node1, node2, path=[], cnt=0):
        #print(cnt)
        if node1 is not None:
            #print(path, node1.data, [n.data for n in node1.adjs])
            #print()
            if node1.data not in seen:
                seen.append(node1.data)
                path += [node1.data]
                q = Queue()
                q.add_list(node1.adjs)
                #q._print()
                while q.size() != 0:
                    node = q.remove()
                    if node == None:
                        pass
                    else:
                        if node.data == node2.data:
                            #print('h', path)
                            path = path+[node2.data]
                            real_path.append([path, len(path)])
                            return path
                        else:
                            _find_route(node, node2, path=path[::], cnt=cnt+1)
    _find_route(node1, node2)
    #print('real', real_path)
    if real_path == []:
        return None
    else:
        sorted_list = sorted(real_path, key=lambda x: x[1])
        out = []
        for e in sorted_list:
            out.append(''.join(e[0]))
        print(set(out))
        return set(out)

class Node():
    def __init__(self, data, adjacency_list=[]):
        self.data = data
        self.adjacency_list = adjacency_list

    @property
    def adjs(self):
        np.random.shuffle(self.adjacency_list)
        return self.adjacency_list

    def add_edge_to(self, node):
        self.adjacency_list += [node]


class Queue():
    def __init__(self):
        self.arr = []

    def add_list(self, items):
        for item in items:
            self.add(item)

    def add(self, item):
        self.arr.append(item)

    def _print(self):
        print([n.data for n in self.arr])

    def remove(self):
        if len(self.arr) > 0:
            first = self.arr[0]
            self.arr = self.arr[1:]
            return first
        else:
            return None

    def size(self):
        return len(self.arr)

import unittest

def str_for(path):
  if not path: return str(path)
  return ''.join([str(n) for n in path])

class Test(unittest.TestCase):
    def test_find_route(self):
        node_j = Node('J')
        node_i = Node('I')
        node_h = Node('H')
        node_d = Node('D')
        node_f = Node('F', [node_i])
        node_b = Node('B', [node_j])
        node_g = Node('G', [node_d, node_h])
        node_c = Node('C', [node_g])
        node_a = Node('A', [node_b, node_c, node_d])
        node_e = Node('E', [node_f, node_a])
        node_d.add_edge_to(node_a)
        self.assertEqual(str_for(find_route(node_a, node_i)), 'None')
        self.assertEqual(find_route(node_a, node_j), {'ABJ'})
        node_h.add_edge_to(node_i)
        self.assertEqual(find_route(node_a, node_i), {'ADI', 'ACGHI', 'ABJI'})

if __name__ == "__main__":
    unittest.main()
