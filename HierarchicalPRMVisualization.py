# coding: utf-8
"""
Enhanced visualization for hierarchical PRM
"""
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def hierarchicalPRMVisualize(planner_result, ax=None, nodeSize=300):
    """Visualize hierarchical PRM result (success or failure)"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    graph = planner_result['roadmap']
    pos = nx.get_node_attributes(graph, 'pos')
    color = nx.get_node_attributes(graph, 'color')

    # Draw main graph
    nx.draw_networkx_nodes(graph, pos, ax=ax,
                          nodelist=list(color.keys()),
                          node_color=list(color.values()),
                          node_size=nodeSize)
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.3)

    # Draw solution path if exists
    if planner_result['path']:
        Gsp = nx.subgraph(graph, planner_result['path'])
        nx.draw_networkx_edges(Gsp, pos, ax=ax, alpha=0.8,
                             edge_color='g', width=3.0)

    # Highlight nodes
    for node in ["start", "goal"]:
        if node in graph.nodes():
            node_type = graph.nodes[node].get('nodeType', 'Terminal')
            color = '#ff9999' if node_type == 'FailedTerminal' else '#00dd00'
            nx.draw_networkx_nodes(graph, pos, nodelist=[node],
                                 node_size=nodeSize * 1.5,
                                 node_color=color, ax=ax)
            nx.draw_networkx_labels(graph, pos,
                                  labels={node: node[0].upper()}, ax=ax)

    # Optional: Draw internal planner attempts if available
    if 'internal_planner_stats' in planner_result:
        for attempt in planner_result['internal_planner_stats'][-3:]:
            internal_pos = nx.get_node_attributes(attempt['graph'], 'pos')
            if internal_pos:
                nx.draw_networkx_nodes(attempt['graph'], internal_pos, ax=ax,
                                     node_size=nodeSize // 2,
                                     node_color='orange', alpha=0.5)
                nx.draw_networkx_edges(attempt['graph'], internal_pos, ax=ax,
                                     edge_color='orange', alpha=0.3)

    # Add title with status
    status = "SUCCESS" if planner_result['path'] else "FAILURE"
    ax.set_title(f"Planning Result: {status}\n"
                 f"Nodes: {len(graph.nodes())} | "
                 f"Edges: {len(graph.edges())} | "
                 f"Components: {nx.number_connected_components(graph)}")

    return ax


def animateHierarchicalSolution(planner_result, environment):
    """Animate hierarchical planning solution"""
    fig = plt.figure(figsize=(14, 7))

    # Main planner view
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Main Planner (Visibility-PRM)")

    # Internal planner view
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Internal Planner Attempts")

    # Get solution path if exists
    path = planner_result.get('path', [])
    graph = planner_result['roadmap']

    def animate(i):
        ax1.clear()
        ax2.clear()

        # Draw main planner up to current point
        if path and i < len(path):
            partial_solution = path[:i + 1]
            pos = nx.get_node_attributes(graph, 'pos')

            # Draw full graph
            environment.drawObstacles(ax1)
            nx.draw_networkx_nodes(graph, pos, ax=ax1, node_size=50, alpha=0.3)
            nx.draw_networkx_edges(graph, pos, ax=ax1, alpha=0.2)

            # Highlight path
            nx.draw_networkx_nodes(graph, pos, nodelist=partial_solution,
                                   ax=ax1, node_size=70, node_color='green')
            if len(partial_solution) > 1:
                path_edges = list(zip(partial_solution[:-1], partial_solution[1:]))
                nx.draw_networkx_edges(graph, pos, edgelist=path_edges,
                                       ax=ax1, edge_color='green', width=2)

        # Draw internal planner attempts
        if i < len(planner_result['internal_planner_stats']):
            attempt = planner_result['internal_planner_stats'][i]
            internal_pos = nx.get_node_attributes(attempt['graph'], 'pos')

            # Draw internal planner graph
            environment.drawObstacles(ax2)
            if internal_pos:
                nx.draw_networkx_nodes(attempt['graph'], internal_pos, ax=ax2,
                                       node_size=50, node_color='orange')
                nx.draw_networkx_edges(attempt['graph'], internal_pos, ax=ax2,
                                       edge_color='orange', alpha=0.5)

                # Highlight start and goal
                nx.draw_networkx_nodes(attempt['graph'],
                                       {0: attempt['start'], 1: attempt['goal']},
                                       nodelist=[0, 1], ax=ax2,
                                       node_size=70,
                                       node_color=['blue', 'red'])

            # Show attempt status
            status = "SUCCESS" if attempt['success'] else "FAILED"
            ax2.text(0.5, 1.05, f"Attempt {i + 1}: {status}",
                     transform=ax2.transAxes, ha='center')

    frames = max(len(path), len(planner_result['internal_planner_stats']))
    ani = FuncAnimation(fig, animate, frames=frames, interval=500, repeat=False)
    plt.close()
    return HTML(ani.to_jshtml())