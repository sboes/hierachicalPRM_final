import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from shapely import plotting

def basic_prm_visualize(planner, solution=[], ax=None, node_size=300):
    graph = planner.graph.copy()
    pos = nx.get_node_attributes(graph, 'pos')

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Hindernisse zeichnen
    if hasattr(planner, "scene"):
        for shape in planner.scene.values():
            xs, ys = shape.exterior.xy if hasattr(shape, 'exterior') else shape.xy
            ax.fill(xs, ys, color='lightcoral', alpha=0.6)

    # Alle Knoten
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=node_size)
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.3)

    # Größter zusammenhängender Bereich (dashed blau)
    if not nx.is_connected(graph):
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        if components:
            Gcc = graph.subgraph(components[0])
            nx.draw_networkx_edges(Gcc, pos,
                                   edge_color='blue', width=2.5,
                                   style='dashed', alpha=0.5, ax=ax)

    # Lösungspfad
    if solution:
        edges = list(zip(solution[:-1], solution[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=edges,
                               edge_color='green', width=6, ax=ax)

    # Start / Goal
    for label, color in [("start", "#00dd00"), ("goal", "#dd0000")]:
        if label in graph.nodes():
            nx.draw_networkx_nodes(graph, pos, nodelist=[label],
                                   node_size=node_size, node_color=color, ax=ax)
            nx.draw_networkx_labels(graph, pos, labels={label: label[0].upper()}, ax=ax)

    ax.set_aspect("equal")
    ax.set_xlim(planner.limits[0])
    ax.set_ylim(planner.limits[1])
    ax.grid(True)
    ax.set_title("Basic PRM Visualisierung")
    return ax
