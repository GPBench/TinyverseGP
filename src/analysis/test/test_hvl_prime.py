import random

N = 15
TRIAL = 10000


class Node:
    def __init__(self, val):
        self.val = val
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        print("attach node", child.val, "as a child of", self.val)


def is_leaf(node):
    """ check if a node is a leaf """
    return not node.children


def get_leafs(n: Node, p=None, leafs: list = None) -> list[tuple[Node, Node]]:
    """
    Recursively obtain all leafs from the tree and store them in a list.
    """
    if leafs is None:
        leafs = []

    if is_leaf(n):
        leafs.append((n, p))
    else:
        for c in n.children:
            get_leafs(c, n, leafs)
    return leafs


def get_inner_nodes(n: Node, p=None, inner_nodes: list = None) -> list[tuple[Node, Node]]:
    """
    Recursively obtain all inner nodes from the tree and store them in a list.
    """
    if inner_nodes is None:
        inner_nodes = []

    if not is_leaf(n):
        inner_nodes.append((n, p))
        for c in n.children:
            get_inner_nodes(c, n, inner_nodes)
    return inner_nodes


def rnd_inner_node(n: Node):
    """
    Select and return a node from the set of inner nodes uniformly at random.
    """
    return random.choice(get_inner_nodes(n))


def rnd_leaf(n: Node):
    """
    Select and return a node from the set of leafs uniformly at random.
    """
    return random.choice(get_leafs(n))


def get_nodes(n: Node, p=None, nodes: list = None) -> list[tuple[Node, Node]]:
    """
    Recursively obtain all nodes from the tree and store them in a list.
    """
    if nodes is None:
        nodes = []

    nodes.append((n, p))

    if not is_leaf(n):
        for c in n.children:
            get_nodes(c, n, nodes)

    return nodes


def rnd_node(n: Node):
    return random.choice(get_nodes(n))


def height(root: Node, d: int = 0):  # from DepthUnbiasedHVLPrime (renamed)
    """
    Calculates the height or maximum depth of a tree.
    """
    if root is None:
        return -1
    if is_leaf(root):  # cuong: changed to adapt to the simpler structure
        return d
    return max([height(c) + 1 for c in root.children])


def size(n: Node) -> int:
    """
    Recursively calculate the number of nodes in the tree.
    """
    if n is None:
        return 0
    if is_leaf(n):
        return 1
    return 1 + sum(size(c) for c in n.children)


## NEW FUNCTIONS

def get_nodes_at_depth(root: Node, depth: int, nodes: list = None, d: int = 0) -> list[tuple[Node, Node]]:
    """
    Recursively selects and returns all nodes at a specified depth.
    """
    if nodes is None:
        nodes = []

    if depth == 0:
        return [(root, None)]

    if d == depth - 1:
        for c in root.children:
            nodes.append((c, root))

    for c in root.children:
        get_nodes_at_depth(c, depth, nodes, d + 1)

    return nodes


def select_node_at_depth(n: Node, d: int) -> tuple[Node, Node]:
    """
    Selects and returns a node at a specified depth uniformly at random.
    """
    nodes = get_nodes_at_depth(n, d)
    return random.choice(nodes) if len(nodes) > 0 else (None, None)


print("create the test tree at root 0")
n = [Node(i) for i in range(N)]
n[0].add_child(n[1])
n[0].add_child(n[2])
n[1].add_child(n[3])
n[1].add_child(n[4])
n[2].add_child(n[5])
n[2].add_child(n[6])
n[4].add_child(n[7])
n[4].add_child(n[8])
n[5].add_child(n[9])
n[5].add_child(n[10])
n[6].add_child(n[11])
n[6].add_child(n[12])
n[8].add_child(n[13])
n[8].add_child(n[14])
print()

print("-- BASIC INFO --")
print()

print("print heights of various nodes")
print("height(None) is", height(None))
print("height(0) is", height(n[0]))
print("height(2) is", height(n[2]))
print("height(4) is", height(n[4]))
print()

print("print sizes of various subtrees")
print("size(None) is", size(None))
print("size(0) is", size(n[0]))
print("size(2) is", size(n[2]))
print("size(4) is", size(n[4]))
print()

print("print all nodes with their parents in format parent<-node")
ret = get_nodes(n[0])
for node, parent in ret:
    print(parent.val if parent is not None else "None", "<-", node.val)
print()

print("print all leaves with their parents in format parent<-node")
ret = get_leafs(n[0])
n_leaf = len(ret)
for node, parent in ret:
    print(parent.val if parent is not None else "None", "<-", node.val)
print()

print("print all internal nodes with their parents in format parent<-node")
ret = get_inner_nodes(n[0])
n_internal = len(ret)
for node, parent in ret:
    print(parent.val if parent is not None else "None", "<-", node.val)
print()

print("-- START TESTING ROMAN's IMPLEMENTATION --")
print()

print("sample a leaf uniformly at random", TRIAL, "times")
stat, stat[None] = {i: 0 for i in range(N)}, 0
for i in range(TRIAL):
    node, parent = rnd_leaf(n[0])
    stat[node.val if node is not None else None] += 1
print("distribution in format node/rate is")
for i in stat.keys():
    print("{}/{}".format(i, stat[i] / TRIAL))
print()

print("sample an internal node uniformly at random", TRIAL, "times")
stat, stat[None] = {i: 0 for i in range(N)}, 0
for i in range(TRIAL):
    node, parent = rnd_inner_node(n[0])
    stat[node.val if node is not None else None] += 1
print("distribution in format node/rate is")
for i in stat.keys():
    print("{}/{}".format(i, stat[i] / TRIAL))
print()

print("sample a node uniformly at random", TRIAL, "times")
stat, stat[None] = {i: 0 for i in range(N)}, 0
for i in range(TRIAL):
    node, parent = rnd_node(n[0])
    stat[node.val if isinstance(node, Node) else None] += 1  # due to bug, the key checking differs
print("distribution in format node/rate is")
for i in stat.keys():
    print("{}/{}".format(i, stat[i] / TRIAL))
print()

print("sample a node from a depth chosen uniformly at random", TRIAL, "times")
h = height(n[0])
stat, stat[None] = {i: 0 for i in range(N)}, 0
for i in range(TRIAL):
    rd = random.randint(0, h)
    node, parent = select_node_at_depth(n[0], rd)
    stat[node.val if node is not None else None] += 1
print("rate of selecting node 0 is", stat[0] / TRIAL)
print("rate of selecting node 1 or 2 is", (stat[1] + stat[2]) / TRIAL)
print("rate of selecting node 13 or 14 is", (stat[13] + stat[14]) / TRIAL)
print("rate of selecting nothing is", (stat[None]) / TRIAL)
print("distribution in format node/rate is")
for i in stat.keys():
    print("{}/{}".format(i, stat[i] / TRIAL))
print()
