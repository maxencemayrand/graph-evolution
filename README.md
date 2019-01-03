# Graph Evolution

Watch a graph (i.e. a network) evolve in time like a classical mechanical system. The nodes are interpreted as masses endowed with 1/r^2 repulsive forces and the edges as attractive springs. By letting the graph move according to the usual laws of physics and with a bit a friction, the graph stabilizes into a nice equilibrium position which can be used to have a visually attractive display of the graph. There is a function to output the final result in LaTeX form.

![Example](./examples/ex2.gif)

## Example

See [this notebook](https://nbviewer.jupyter.org/github/maxencemayrand/graph-evolution/blob/master/nilpotent-orbits.ipynb) for an example where `graphevolution` is used to get a neat image of a partially ordered set that occurs naturally in Lie theory.

## Usage

To see the program in action with a simple example, generate a random graph as follows (in Python).
```python
import graphevolution as ge
import matplotlib.pyplot as plt
g = ge.random()
g.evolve()
plt.show()
```
The last line displays a matplotlib animation.

To see your own graph evolve, you can initiate a Graph object by a list of edges, where each edge is a tuple of 2 nodes. A node can be any python object; they are just labels. For example:
```python
import graphevolution as ge
import matplotlib.pyplot as plt
g = ge.Graph()
edges = [('A', 'B'), ('B', 'C'), ('B', 'D')]
g.set_by_edges(edges)
g.evolve()
plt.show()
```
Once the graph is in a nice configuration, you can output it in LaTeX form. For instance, with the example above,
```python
g.latex()
```
outputs
```
$$
\begin{tikzpicture}
\tikzset{dot/.style={draw, circle, fill, inner sep=1pt},}
\node[dot] ('B') at 	(0.223, -0.344) {};
\node[dot] ('A') at 	(1.369, -0.539) {};
\node[dot] ('C') at 	(-0.519, -1.242) {};
\node[dot] ('D') at 	(-0.188, 0.746) {};
\draw ('B') -- ('C');
\draw ('B') -- ('D');
\draw ('A') -- ('B');
\end{tikzpicture}
$$
```

Then, copy and paste this into a LaTeX document with `\usepackage{tikz}` in the preamble.


The `evolve` function has many parameters; its signature is
```python
Graph.evolve(friction=0.5, tension=1.0, repulsion=1.0, magnetic=False,
   dt=0.01, dim=4, nodesize=5, edgesize=1, repeat=10, interval=1, frames=500)
```
The most important ones are:

* `friction`: The amount of friction that the nodes are subject to.
* `tension`: The amount of tension in the springs (i.e. the edges).
* `repulsion`: The amount of repulsive force that each node applies to the others.
* `magnetic`: Whether the nodes tend to align on a North-South axis (by a magnetic force like a compass needle).

See the source code for the other parameters.
