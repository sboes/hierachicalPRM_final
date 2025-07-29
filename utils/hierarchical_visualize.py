# utils/hierarchical_visualize.py

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def visualize_hierarchical_planning(hier_planner, collision_checker, config_low):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    xlim, ylim = hier_planner.limits

    # --- 1. High-Level Roadmap ---
    graph = hier_planner.global_planner.graph
    pos = nx.get_node_attributes(graph, 'pos')
    nx.draw_networkx_nodes(graph, pos, ax=axes[0], node_color='lightblue', node_size=100)
    nx.draw_networkx_edges(graph, pos, ax=axes[0], alpha=0.2)

    if hier_planner.global_path:
        global_edges = list(zip(hier_planner.global_path[:-1], hier_planner.global_path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=global_edges, ax=axes[0], edge_color='green', width=3)

    for label, color in [("start", "green"), ("goal", "red")]:
        if label in graph.nodes:
            nx.draw_networkx_nodes(graph, pos, nodelist=[label], node_color=color, node_size=300, ax=axes[0])
            nx.draw_networkx_labels(graph, pos, labels={label: label[0].upper()}, ax=axes[0])

    if hasattr(collision_checker, "drawObstacles"):
        collision_checker.drawObstacles(axes[0])

    axes[0].set_title("1. High-Level Roadmap")
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    axes[0].set_aspect("equal")
    axes[0].grid(True)

    # --- 2. Alle Low-Level Subplanner anzeigen ---
    axes[1].set_title("2. Low-Level Subplaner (alle)")
    for subplanner, sub_path in hier_planner.get_subplanners():
        sub_pos = nx.get_node_attributes(subplanner.graph, 'pos')
        nx.draw_networkx_nodes(subplanner.graph, sub_pos, ax=axes[1], node_color='skyblue', node_size=20, alpha=0.6)
        nx.draw_networkx_edges(subplanner.graph, sub_pos, ax=axes[1], alpha=0.2)

        path_edges = list(zip(sub_path[:-1], sub_path[1:]))
        nx.draw_networkx_edges(subplanner.graph, sub_pos, edgelist=path_edges,
                               edge_color='green', width=2.5, ax=axes[1])

    if hasattr(collision_checker, "drawObstacles"):
        collision_checker.drawObstacles(axes[1])

    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)
    axes[1].set_aspect("equal")
    axes[1].grid(True)

    # --- 3. Finaler Pfad ---
    if hier_planner.solution_path:
        coords = hier_planner.solution_path
        xs, ys = zip(*coords)
        axes[2].plot(xs, ys, color='blue', linewidth=3)
        axes[2].scatter(xs[0], ys[0], color='green', s=100)
        axes[2].scatter(xs[-1], ys[-1], color='red', s=100)

    if hasattr(collision_checker, "drawObstacles"):
        collision_checker.drawObstacles(axes[2])

    axes[2].set_title("3. Finaler Hierarchischer Pfad")
    axes[2].set_xlim(xlim)
    axes[2].set_ylim(ylim)
    axes[2].set_aspect("equal")
    axes[2].grid(True)

    # alle Achsen gleich skalieren
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(True)
    plt.tight_layout()
    plt.show()