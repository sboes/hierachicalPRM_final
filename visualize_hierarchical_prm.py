
# coding: utf-8
"""
Visualization helper for the Hierarchical PRM.

Run this file directly to see a quick demo scene, or import `plot_hierarchical_result`
from another script / notebook. Requires matplotlib and networkx.
"""

from typing import Dict, Any, List
import matplotlib.pyplot as plt
import networkx as nx

from lectures.IPEnvironment import CollisionChecker
from shapely.geometry import Polygon

from hierarchical_prm import HierarchicalPRM


def plot_graph(ax, graph, **kwargs):
    """Draw a roadmap: nodes and edges."""
    pos = nx.get_node_attributes(graph, 'pos')
    # edges
    for (u, v) in graph.edges():
        if u in pos and v in pos:
            p1, p2 = pos[u], pos[v]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.8)
    # nodes
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.scatter(xs, ys, s=15, zorder=3)


def plot_hierarchical_result(env: CollisionChecker,
                             start: List[float],
                             goal: List[float],
                             main_config: Dict[str, Any],
                             inner: Dict[str, Any]):
    planner = HierarchicalPRM(env, inner_type=inner.get('type', 'basic'))
    cfg = dict(main_config)
    cfg['inner'] = inner

    path = planner.planPath([start], [goal], cfg)

    fig, ax = plt.subplots(figsize=(6, 6))
    # draw obstacles if available
    try:
        env.drawObstacles(ax)
    except Exception:
        pass

    # roadmap
    plot_graph(ax, planner.graph)

    # path if found
    if path:
        pos = nx.get_node_attributes(planner.graph, 'pos')
        xs = [pos[n][0] for n in path]
        ys = [pos[n][1] for n in path]
        ax.plot(xs, ys, linewidth=2.5)

    ax.scatter([start[0]], [start[1]], s=60, marker='*')
    ax.scatter([goal[0]], [goal[1]], s=60, marker='o')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Hierarchical PRM')

    plt.show()
    return path, planner


if __name__ == "__main__":
    # Simple demo scene (two rectangular obstacles)
    scene = {
        'obs1': Polygon([(5, 5), (9, 5), (9, 14), (5, 14)]),
        'obs2': Polygon([(12, 8), (16, 8), (16, 18), (12, 18)]),
    }
    env = CollisionChecker(scene, limits=((0, 22), (0, 22)))

    start = [2.0, 2.0]
    goal = [20.0, 20.0]

    main_cfg = {'ntry': 80}
    inner_cfg = {'type': 'lazy',
                 'config': {
                     'numNodes': 200,
                     'kNN': 10,
                     'maxEdgeLen': 5.0
                 }}

    plot_hierarchical_result(env, start, goal, main_cfg, inner_cfg)
