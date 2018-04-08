import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Graph():
    """
    A finite directed weighted graph.

    Nodes:  They are implemented by the dictionary `nodes`, whose keys
            are labels for the nodes and whose values are weigths.
    Edges:  They are implemented by the set `edges` which consists of
            tuples of two keys in `nodes`.

    Example
    ========
    >>> from graphevolution import Graph
    >>> nodes = {'A': 6, 'B': 4, 'C': 5, 'D': 1}
    >>> edges = set([('A', 'B'), ('B', 'C'), ('B', 'D')])
    >>> g = Graph(nodes, edges)
    >>> g.info()
    <class 'graphevolution.Graph'>

    4 nodes
    3 edges

    Nodes:
    A  :  6
    B  :  4
    C  :  5
    D  :  1

    Edges:
    ('B', 'D') ('B', 'C') ('A', 'B')
    """

    def __init__(self, nodes={}, edges=set()):
        self.nodes = nodes  # dict
        self.edges = edges  # set

    def set_by_sets(self, nodes, edges):
        self.nodes = {n: np.vstack([np.random.normal(0, 0.5, 2), np.zeros(2)])
                        for n in nodes}
        self.edges = edges

    def set_by_edges(self, edges):
        nodes_set = set()
        for e in edges:
            nodes_set.add(e[0])
            nodes_set.add(e[1])
        self.nodes = {n: np.vstack([np.random.normal(0, 0.5, 2), np.zeros(2)])
                        for n in nodes_set}
        self.edges = edges

    def random(self, n_edges, n_nodes):
        edges = set()
        while len(edges) < n_edges:
            n1 = 0; n2 = 0
            while n1 == n2:
                n1 = np.random.choice(range(n_nodes))
                n2 = np.random.choice(range(n_nodes))
            if (n2, n1) not in edges:
                edges.add((n1, n2))
        self.set_by_edges(edges)

    def reset(self):
        self.nodes = {n: np.vstack([np.random.normal(0, 0.5, 2), np.zeros(2)])
                        for n in self.nodes}

    def info(self):
        print("<class 'graphevolution.Graph'>\n")
        print("{} nodes".format(len(self.nodes)))
        print("{} edges\n".format(len(self.edges)))
        print("Nodes:")
        for k, v in self.nodes.items():
            print(k, " : ", v)
        print()
        print("Edges:")
        print(*self.edges)

    def latex(self):
        print("$$")
        print("\\begin{tikzpicture}")
        print("\\tikzset{dot/.style={draw, circle, fill, inner sep=1pt},}")
        for n, c in self.nodes.items():
            print("\\node[dot] (%r) at \t(%.3f, %.3f) {};" % (n, c[0, 0], c[0, 1]))
        for e in self.edges:
            print("\\draw (%r) -- (%r);" % (e[0], e[1]))
        print("\\end{tikzpicture}")
        print("$$")

    def step(self,
                dt=0.01,
                friction=0.5,
                tension=1.0,
                repulsion=1.0,
                magnetic=False):
        F = {n : np.zeros(2) for n in self.nodes}
        for n, c in self.nodes.items():
            for e in self.edges:
                if n in e:
                    v = self.nodes[e[0]][0] - self.nodes[e[1]][0]
                    if n == e[1]:
                        v = -v
                    F[n] += -tension * v
            F[n] += -friction * c[1]
            for m, d in self.nodes.items():
                if m != n:
                    v = c[0] - d[0]
                    nor = np.linalg.norm(v)
                    F[n] += repulsion * v / (nor ** 3)
        if magnetic != False:
            for e in self.edges:
                F[e[0]] += np.array([0, -magnetic])
                F[e[1]] += np.array([0, +magnetic])
        for n, c in self.nodes.items():
            p = F[n] * (dt ** 2) / 2 + c[1] * dt + c[0]
            v = F[n] * dt + c[1]
            self.nodes[n] = np.vstack([p, v])
        return F

    def plot(self):
        fig, ax = plt.subplots()
        ax.axis('equal')
        ax.axis('off')
        ax.set( xlim=(-2, 2),
                ylim=(-2, 2))
        for n in self.nodes:
            ax.plot(self.nodes[n][0, 0], self.nodes[n][0, 1], 'ok')
        for e in self.edges:
            p = np.vstack([self.nodes[e[0]][0], self.nodes[e[1]][0]])
            ax.plot(p.T[0], p.T[1], 'k')
        plt.show()

    def anim(self,
                dim=4,
                dt=0.01,
                friction=0.5,
                tension=1.0,
                repulsion=1.0,
                nodesize=5,
                edgesize=1,
                magnetic=False,
                repeat=10):
        L = list(self.edges)

        fig, ax = plt.subplots()
        ax.axis('equal')
        ax.axis('off')
        ax.set( xlim=(-dim, dim),
                ylim=(-dim, dim));

        lines = []

        def init():
            for e in L:
                p = np.vstack([self.nodes[e[0]][0], self.nodes[e[1]][0]])
                lines.append(ax.plot(p.T[0], p.T[1], '-ok',
                                markersize=nodesize,
                                linewidth=edgesize)[0])
            return tuple(lines)

        def update(j):
            for i in range(repeat):
                self.step(
                            dt,
                            friction,
                            tension,
                            repulsion,
                            magnetic
                        )
            for i, e in enumerate(L):
                p = np.vstack([self.nodes[e[0]][0], self.nodes[e[1]][0]])
                lines[i].set_data(p.T[0], p.T[1])
            return tuple(lines)

        ani = animation.FuncAnimation(fig,
                                      update,
                                      interval=1,
                                      init_func=init,
                                      blit=True)

        plt.show()

def example(e=20, n=10, d=4):
    G = Graph()
    G.random(e, n)
    G.anim(dim=d)
    return G
