import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import networkx as nx
from utils.planarRobot import interpolate_line

def animate_planar_path(robot, planner, path, ax, title=""):
    """
    Visualisiert eine animierte Bewegung des Planarroboters entlang eines geplanten Pfades.
    - `robot`: Instanz von PlanarRobot
    - `planner`: z.B. BasicPRM mit Graph
    - `path`: Liste von Knotennummern
    - `ax`: Matplotlib-Achse für die Darstellung
    """
    pos_map = nx.get_node_attributes(planner.graph, 'pos')
    path_coords = [pos_map[node] for node in path]

    # Interpolierter Pfad für flüssigere Animation
    smooth_path = [path_coords[0]]
    for i in range(1, len(path_coords)):
        seg = interpolate_line(path_coords[i - 1], path_coords[i], step_l=0.1)
        smooth_path.extend(seg[1:])

    fig = ax.figure

    def update(i):
        ax.clear()
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title(f"{title} (Frame {i + 1}/{len(smooth_path)})")
        ax.grid(True)

        # Roboter bewegen
        robot.move(smooth_path[i])
        joint_positions = robot.get_transforms()

        # Robotersegmente zeichnen
        for j in range(1, len(joint_positions)):
            xs = [joint_positions[j - 1][0], joint_positions[j][0]]
            ys = [joint_positions[j - 1][1], joint_positions[j][1]]
            ax.plot(xs, ys, 'go-', linewidth=2)

    ani = animation.FuncAnimation(fig, update, frames=len(smooth_path), interval=100)
    return HTML(ani.to_jshtml())
