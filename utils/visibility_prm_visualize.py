# utils/visibility_prm_visualize.py
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString


def visibility_prm_visualize(planner, solution=[], ax=None, node_size=300):
    graph = planner.graph
    statsHandler = getattr(planner, "statsHandler", None)
    pos = nx.get_node_attributes(graph, 'pos')
    color = nx.get_node_attributes(graph, 'color')

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Hindernisse
    if hasattr(planner, "scene"):
        for shape in planner.scene.values():
            xs, ys = shape.exterior.xy if hasattr(shape, 'exterior') else shape.xy
            ax.fill(xs, ys, color='lightcoral', alpha=0.6)
    elif hasattr(planner, "_collisionChecker") and hasattr(planner._collisionChecker, "drawObstacles"):
        planner._collisionChecker.drawObstacles(ax)

    # Visualisierung der getesteten Sichtbarkeiten (statHandler)
    if statsHandler:
        sgraph = statsHandler.graph
        spos = nx.get_node_attributes(sgraph, 'pos')
        nx.draw_networkx_nodes(sgraph, pos=spos, alpha=0.2, node_size=node_size)
        nx.draw_networkx_edges(sgraph, pos=spos, alpha=0.2, edge_color='yellow')

    # Sichtbarkeitsgraph
    if color:
        nx.draw_networkx_nodes(graph, pos, ax=ax,
                               nodelist=list(color.keys()),
                               node_color=list(color.values()),
                               node_size=node_size)
    else:
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=node_size)

    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.3)

    # Größte Komponente hervorheben
    if not nx.is_connected(graph):
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        if components:
            Gcc = graph.subgraph(components[0])
            nx.draw_networkx_edges(Gcc, pos, edge_color='blue',
                                   width=2.0, style='dashed', alpha=0.5, ax=ax)

    # Lösungspfad (grün)
    if solution:
        Gsp = graph.subgraph(solution)
        nx.draw_networkx_edges(Gsp, pos, edge_color='green',
                               width=5.0, alpha=0.8, ax=ax, label="Solution Path")

    # Start / Ziel
    for label, color in [("start", "#00dd00"), ("goal", "#dd0000")]:
        if label in graph.nodes():
            nx.draw_networkx_nodes(graph, pos, nodelist=[label],
                                   node_size=node_size, node_color=color, ax=ax)
            nx.draw_networkx_labels(graph, pos, labels={label: label[0].upper()}, ax=ax)

    ax.set_aspect("equal")
    ax.set_xlim(planner.limits[0])
    ax.set_ylim(planner.limits[1])
    ax.grid(True)
    ax.set_title("Visibility PRM Visualisierung")
    return ax
