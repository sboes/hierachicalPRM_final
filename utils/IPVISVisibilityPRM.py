import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def custom_visibility_prm_visualize(prm, path, ax, draw_graph=True, draw_stats=False):
    """
    Zeichnet:
      - Hindernisse aus dem CollisionChecker
      - Visibility-Graph (Knoten + Kanten), optional
      - Subplanner-Tests (statsHandler graph), optional
      - Finalen, kollisionsfreien Pfad
    """
    # Hindernisse
    try:
        prm._collisionChecker.drawObstacles(ax)
    except AttributeError:
        if hasattr(prm._collisionChecker, 'scene'):
            for poly in prm._collisionChecker.scene:
                try:
                    if hasattr(poly, 'exterior'):
                        xs, ys = zip(*poly.exterior.coords)
                    else:
                        pts = np.asarray(poly)
                        xs, ys = pts[:,0], pts[:,1]
                    ax.fill(xs, ys, color='red', alpha=0.5, zorder=0)
                except Exception:
                    continue

    # Visibility-Roadmap
    if draw_graph:
        pos = nx.get_node_attributes(prm.graph, 'pos')
        if pos:
            xs, ys = zip(*pos.values())
            ax.scatter(xs, ys, s=20, c='lightblue', alpha=0.5, zorder=2)
            for u, v in prm.graph.edges():
                p1, p2 = pos[u], pos[v]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        lw=0.5, c='gray', alpha=0.3, zorder=1)

    # Subplanner-Test-Kanten
    if draw_stats and getattr(prm, 'statsHandler', None) is not None:
        stats_graph = prm.statsHandler.graph
        pos_stats = nx.get_node_attributes(stats_graph, 'pos')
        for u, v in stats_graph.edges():
            if u in pos_stats and v in pos_stats:
                p1, p2 = pos_stats[u], pos_stats[v]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        lw=0.5, c='orange', alpha=0.4, zorder=1)

    # Gefundener Pfad
    if path:
        coords = prm.get_path_coordinates(path)
        if coords:
            xs, ys = zip(*coords)
            ax.plot(xs, ys, lw=3, c='lime', zorder=4, label='Solution Path')

    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()


def visualize_prm_three_views(prm, path, figsize=(18,6)):
    """
    Erzeugt drei Subplots:
      1) Nur alle generierten Visibility-Knoten
      2) Nodes mit Subplanner-Test-Kanten
      3) Finaler, kollisionsfreier Pfad
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    titles = ['All Nodes', 'Subplanner Tests', 'Final Path']
    for ax, title in zip(axs, titles):
        draw_graph = True if title != 'Final Path' else False
        draw_stats = (title == 'Subplanner Tests')
        custom_visibility_prm_visualize(prm, path, ax,
                                        draw_graph=draw_graph,
                                        draw_stats=draw_stats)
        ax.set_title(title)
    plt.tight_layout()
    return fig, axs


#--------------------------------------------------
# Benchmark-Plot-Funktion
#--------------------------------------------------

def plot_benchmark_results(df, scene_column='scene'):
    """
    Erzeugt pro Szene Balkendiagramme für Pfadknoten, euklidische Länge,
    Laufzeit und Roadmap-Größe.
    df: DataFrame mit Spalten ['scene', 'path_nodes', 'euclid_length', 'time_s', 'roadmap_size']
    """
    for scene in df[scene_column].unique():
        d = df[df[scene_column] == scene]
        fig, ax1 = plt.subplots(figsize=(10, 6))
        width = 0.2
        idx = np.arange(len(d))

        ax1.bar(idx, d['path_nodes'], width, label='Pfad-Knoten', alpha=0.8)
        ax1.set_ylabel('Pfad-Knoten')
        ax1.set_xticks(idx)
        ax1.set_xticklabels(d[scene_column])

        ax2 = ax1.twinx()
        ax2.bar(idx + width, d['euclid_length'], width, label='Euklid.-Länge', alpha=0.8)
        ax2.set_ylabel('Euklid.-Länge')

        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('axes', 1.2))
        ax3.bar(idx + 2*width, d['time_s'], width, label='Zeit [s]', alpha=0.8)
        ax3.set_ylabel('Zeit [s]')

        ax4 = ax1.twinx()
        ax4.spines['right'].set_position(('axes', 1.4))
        ax4.bar(idx + 3*width, d['roadmap_size'], width, label='Roadmap Größe', alpha=0.8)
        ax4.set_ylabel('Roadmap Größe')

        ax1.set_title(f'Benchmark-Auswertung: {scene}')
        handles, labels = [], []
        for ax in (ax1, ax2, ax3, ax4):
            h, l = ax.get_legend_handles_labels()
            handles += h; labels += l
        ax1.legend(handles, labels, loc='upper left', fontsize='small')

        plt.tight_layout()
        plt.show()
