# === PRM-Erweiterung für n-DoF Roboter mit echtem Kinematikmodell und kollisionsfreier Pfadanalyse ===

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapely.geometry import LineString
from lectures.IPEnvironmentKin import KinChainCollisionChecker
from lectures.IPPlanarManipulator import PlanarRobot
from lectures.IPBasicPRM import BasicPRM


# --- Hilfsfunktion zur Visualisierung ---
def visualize_path(robot, scene, path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_aspect('equal')
    ax.grid(True)

    for shape in scene.values():
        xs, ys = shape.exterior.xy if hasattr(shape, 'exterior') else shape.xy
        ax.fill(xs, ys, color='lightcoral', alpha=0.7)

    for config in path:
        robot.move(config)
        positions = robot.get_transforms()
        for i in range(1, len(positions)):
            x0, y0 = positions[i - 1]
            x1, y1 = positions[i]
            ax.plot([x0, x1], [y0, y1], '-o', color='blue')

    plt.title("Pfad im Arbeitsraum")
    plt.show()


# --- Animation des geplanten Pfads ---
def animate_solution(robot, path, scene):
    fig, ax = plt.subplots()
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_aspect('equal')
    ax.grid(True)
    for shape in scene.values():
        xs, ys = shape.exterior.xy if hasattr(shape, 'exterior') else shape.xy
        ax.fill(xs, ys, color='lightcoral', alpha=0.7)

    line, = ax.plot([], [], '-o', color='green')

    def update(i):
        robot.move(path[i])
        positions = robot.get_transforms()
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        line.set_data(xs, ys)
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(path), interval=500, blit=True)
    plt.show()


# --- PRM-Ausführung mit realem Roboter im Konfigurationsraum ---
def run_ndof_prm(dof=4):
    robot = PlanarRobot(n_joints=dof)
    scene = {
        "obstacle1": LineString([(-1.5, 0), (1.5, 0)]).buffer(0.3),
        "obstacle2": LineString([(0, 1.5), (0, -1.5)]).buffer(0.2)
    }
    limits = [
        np.array(np.full((dof,), -np.pi), dtype=np.float32),
        np.array(np.full((dof,), np.pi), dtype=np.float32)
    ]

    assert limits[0].shape == (dof,), f"limits[0] falsch geformt: {limits[0].shape}"
    assert limits[1].shape == (dof,), f"limits[1] falsch geformt: {limits[1].shape}"

    checker = KinChainCollisionChecker(robot, scene, limits=limits, fk_resolution=0.05)
    planner = BasicPRM(checker)

    start_cfg = [0.0 for _ in range(dof)]
    goal_cfg = [np.pi / 4 for _ in range(dof)]

    config = {"radius": 3.0, "numNodes": 300}
    path = planner.planPath([start_cfg], [goal_cfg], config)



# --- Ausführen ---
if __name__ == '__main__':
    for dof in [2, 4, 6]:
        print(f"\n🔧 Starte PRM-Test mit {dof} DoF")
        run_ndof_prm(dof)
