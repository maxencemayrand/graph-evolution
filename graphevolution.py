import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Graph():
    """
    A finite directed graph in motion in a 2D plane.

    The nodes are implemented by a dictionary `nodes`, whose keys
    are labels for the nodes and whose values are the coordinates of
    the nodes (in the plane) plus their velocities. Each value is a
    two by two numpy array: the first row are the coordinates of the
    node and the second the coordinates of its velocity.

    The edges are implemented by a set `edges` which consists of tuples of
    two keys in `nodes`.

    Example
    ========
    >>> import numpy as np
    >>> import graphevolution as ge
    >>>
    >>> nodeA = np.array([[1, 2], [0, 1]])
    >>> nodeB = np.array([[-1, 0], [3, 2]])
    >>> nodeC = np.array([[2, 1], [2, 2]])
    >>> nodes = {'A': nodeA, 'B': nodeB, 'C': nodeC}
    >>> edges = set([('A', 'B'), ('B', 'C')])
    >>> g = ge.Graph(nodes, edges)
    """

    def __init__(self, nodes={}, edges=set()):
        self.nodes = nodes  # dict
        self.edges = edges  # set

    def set_by_edges(self, edges):
        """
        Initiate the nodes and edges with random coordinates and zero velocities
        by passing only a set of edges. The set `edges` can be any set of
        2-tuples. The elements of these tuples will be the labels of the nodes.

        Parameters
        ==========
        edges: a set of 2-tuples.

        Example
        =======
        >>> import graphevolution as ge
        >>> edges = set([('A', 'B'), ('B', 'C'), ('B', 'D')])
        >>> g = ge.Graph()
        >>> g.set_by_edges(edges)
        >>> g.evolve()
        """
        nodes_set = set()
        for e in edges:
            nodes_set.add(e[0])
            nodes_set.add(e[1])
        self.nodes = {n: np.vstack([np.random.randn(2), np.zeros(2)])
                        for n in nodes_set}
        self.edges = set(edges)

    def reset(self):
        """
        Reset the coordinates of the nodes randomly and set the velocities to
        zero.
        """
        self.nodes = {n: np.vstack([np.random.randn(2), np.zeros(2)])
                        for n in self.nodes}

    def info(self, listnodes=True, listedges=True):
        """
        Print basic info of the Graph on the terminal.

        Parameters
        ==========
        listnodes: If True, will print all labels of the nodes.
        listedges: If True, will print all edges.
        """
        print("<class 'graphevolution.Graph'>\n")
        print("{} nodes".format(len(self.nodes)))
        print("{} edges\n".format(len(self.edges)))
        if listnodes:
            print("Nodes:")
            for k, v in self.nodes.items():
                print(k, end=', ')
        print()
        if listedges:
            print("Edges:")
            print(*self.edges)

    def latex(self):
        """
        Output a LaTeX version of the Graph.
        Copy-paste to your LaTeX document with the `tikz` package   in your
        preamble.
        """
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
        """
        Evolve the graph by one step. The nodes are interpreted as masses of
        equal weights, the edges as springs and they evolve according to the
        usual laws of classical mechanics.

        Parameters
        ==========
        dt: Time interval to integrate the equation of motion.
        friction: The friction coefficient for the nodes.
        tension: The tension in the edges.
        repulsion: If positive, each node has a repulsive ``1 / r**2`` force
            which acts on the other nodes.
        magnetic: If positive, the nodes tend to align in the North-South axis
            and pointing to the North with respect to their orientation.
        """
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

    def evolve(self,
                dim=4,
                dt=0.01,
                friction=0.5,
                tension=1.0,
                repulsion=1.0,
                nodesize=5,
                edgesize=1,
                magnetic=False,
                repeat=10,
                interval=1,
                frames=500):
        """
        Let the graph evolve in time according to the laws of classical
        mechanics. The nodes are interpreted as masses of equal weights and the
        edges as springs.

        The function displays a matplotlib animation.

        Parameters
        ==========
        dt: Time interval to integrate the equation of motion.
        friction: The friction coefficient for the nodes.
        tension: The tension in the edges.
        repulsion: If positive, each node has a repulsive ``1 / r**2`` force
            which acts on the other nodes.
        magnetic: If positive, the nodes tend to align in the North-South axis
            and pointing to the North with respect to their orientation.
        """
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
                                      interval=interval,
                                      init_func=init,
                                      frames=frames,
                                      blit=True)

        return ani

    def copy(self):
        """
        Return a copy of the Graph
        """
        nodes = self.nodes.copy()
        edges = self.edges.copy()
        g = Graph(nodes, edges)
        return g


def random(n_edges=20, n_nodes=10):
    """
    Return a Graph object with `n_nodes` nodes and `n_edges` edges set randomly.
    The coordinates of the nodes are random and the velocities are zero.
    """
    edges = set()
    while len(edges) < n_edges:
        n1 = 0; n2 = 0
        while n1 == n2:
            n1 = np.random.choice(range(n_nodes))
            n2 = np.random.choice(range(n_nodes))
        if (n2, n1) not in edges:
            edges.add((n1, n2))
    g = Graph()
    g.set_by_edges(edges)
    return g
