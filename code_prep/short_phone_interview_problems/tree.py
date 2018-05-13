'''
Binary Search Tree Algorithm
'''

def test():
    l = [10, 5, 6, 3, 8, 2, 1, 11, 9, 4]
    tree = Tree()
    for i in l:
        tree.add(i)

    assert sorted(tree.get_tree_values()) == sorted(l)

    assert tree.root.left.right.right.right.value == 9
    assert tree.root.left.left.value == 3

    # Searching for nodes 16 and 6
    assert tree.search(16) == False
    assert tree.search(3) == True


    ''' Print also
        10
       5 11
      3 6
     2 4 8
    1     9
    '''
    assert tree.print_tree()['4'] == (-1, 3)

    assert tree.preorder() == [10, 5, 3, 2, 1, 4, 6, 8, 9, 11]
    assert tree.inorder() == [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
    assert tree.postorder() == [1, 2, 4, 3, 9, 8, 6, 5, 11, 10]
    return tree

class Node:

    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def _add(self, value):
        if value < self.value:
            if self.left == None:
                self.left = Node(value)
            else:
                self.left._add(value)
        else:
            if self.right == None:
                self.right = Node(value)
            else:
                self.right._add(value)

def info(msg, var):
    print(msg, var)

class Tree:

    def __init__(self):
        self.root = None
        self.cnt = 0

    def add(self, value):
        if self.root == None:
            self.root = Node(value)
        else:
            self.root._add(value)

    def get_tree_values(self):
        L = []
        self._print_tree_values(self.root, L)
        return L

    def _print_tree_values(self, node, L):
        if node == None:
            pass
        else:
            L.append(node.value)
            if node.left is not None:
                self._print_tree_values(node.left, L)
            if node.right is not None:
                self._print_tree_values(node.right, L)

    def print_tree(self):
        d = {}
        self._create_dict(self.root, d, 0, 0)
        self.prin_d(d)
        return d

    def prin_d(self, d):
        used_key = []
        keys = d.keys()
        depth = 0
        holder = []
        while sorted(used_key) != sorted(keys):
            sentence = ''
            sub_list = []
            for key in keys:
                if d[key][1] == depth:
                    used_key.append(key)
                    sub_list.append([int(key), d[key][0]])
            holder.append(sorted(sub_list, key=lambda x:x[1]))
            depth+=1
        min, max = holder[-1][0][1], holder[-1][-1][1]
        for e in holder:
            h = [' ' for _ in range(max-min+1)]
            for sub_e in e:
                h[sub_e[1]-min] = str(sub_e[0])
            print(''.join(h))

    def _create_dict(self, node, d, index, depth):
        if node is not None:
            d[str(node.value)] = (index, depth)
            self._create_dict(node.left, d, index -1, depth+1)
            self._create_dict(node.right, d, index + 1, depth+1)

    def search(self, value):
        return self._search(self.root, value)

    def _search(self, node, value):
        if node == None:
            return False
        node_value = node.value
        if node_value < value:
            return self._search(node.right, value)
        elif node_value > value:
            return self._search(node.left, value)
        else:
            return True

    def preorder(self):
        L = []
        def _preorder(node, L):
            if node is not None:
                L.append(node.value)
                if node.left is not None:
                    _preorder(node.left, L)
                if node.right is not None:
                    _preorder(node.right, L)
        _preorder(self.root, L)
        return L

    def postorder(self):
        L = []
        def _postorder(node, L):
            if node is not None:
                if node.left is not None:
                    _postorder(node.left, L)
                if node.right is not None:
                    _postorder(node.right, L)
                L.append(node.value)
        _postorder(self.root, L)
        return L

    def inorder(self):
        L = []
        def _inorder(node, L):
            if node is not None:
                if node.left is not None:
                    _inorder(node.left, L)
                L.append(node.value)
                if node.right is not None:
                    _inorder(node.right, L)
        _inorder(self.root, L)
        return L
