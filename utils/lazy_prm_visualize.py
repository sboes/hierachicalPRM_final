# utils/lazy_prm_visualize.py

import networkx as nx
import matplotlib.pyplot as plt

def lazy_prm_visualize(planner, solution=[], ax=None, node_size=300):
    graph = planner.graph.copy()
    pos = nx.get_node_attributes(graph, 'pos')

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Hindernisse zeichnen
    if hasattr(planner, "scene"):
        for shape in planner.scene.values():
            xs, ys = shape.exterior.xy if hasattr(shape, 'exterior') else shape.xy
            ax.fill(xs, ys, color='lightcoral', alpha=0.6)

    node_colors = nx.get_node_attributes(graph, 'color')
    if node_colors:
        nx.draw_networkx_nodes(graph, pos, ax=ax,
                               nodelist=list(node_colors.keys()),
                               node_color=list(node_colors.values()),
                               node_size=node_size)
    else:
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=node_size)

    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.2)

    if not nx.is_connected(graph):
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        if components:
            Gcc = graph.subgraph(components[0])
            nx.draw_networkx_edges(Gcc, pos,
                                   edge_color='blue', width=2.5,
                                   style='dashed', alpha=0.5, ax=ax)

    if hasattr(planner, 'colliding_edges') and planner.colliding_edges:
        nx.draw_networkx_edges(graph, pos,
                               edgelist=planner.colliding_edges,
                               edge_color='red', width=4, alpha=0.6, ax=ax)

    if hasattr(planner, 'non_colliding_edges') and planner.non_colliding_edges:
        nx.draw_networkx_edges(graph, pos,
                               edgelist=planner.non_colliding_edges,
                               edge_color='gold', width=3, alpha=0.8, ax=ax)

    if solution:
        path_edges = list(zip(solution[:-1], solution[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges,
                               edge_color='green', width=6, ax=ax)

    for label, color in [("start", "#00dd00"), ("goal", "#dd0000")]:
        if label in graph.nodes():
            nx.draw_networkx_nodes(graph, pos, nodelist=[label],
                                   node_size=node_size, node_color=color, ax=ax)
            nx.draw_networkx_labels(graph, pos, labels={label: label[0].upper()}, ax=ax)

    ax.set_aspect("equal")
    ax.set_xlim(planner.limits[0])
    ax.set_ylim(planner.limits[1])
    ax.grid(True)
    ax.set_title("Lazy PRM Visualisierung")
    return ax
