import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import sympy as sp
from lectures.IPPlanarManipulator import PlanarRobot
from lectures.IPEnvironmentKin import KinChainCollisionChecker


def generate_consistent_joint_config(dim, total_angle=0.0, curvature=0.0):
    """Generiert eine konsistente Gelenkwinkelkonfiguration."""
    base = np.ones(dim) * (total_angle / dim)
    curve = np.linspace(-curvature, curvature, dim)
    return base + curve


def planarRobotVisualize(robot, ax, color='blue'):
    """Visualisiert den planaren Roboter."""
    # statt getEndpointPositions() verwenden wir die in PlanarRobot definierte Methode:
    points = robot.get_transforms()
    x_coords = [float(point[0]) for point in points]
    y_coords = [float(point[1]) for point in points]
    ax.plot(x_coords, y_coords, '-o', color=color)



def setup(n_dof: int, obst=None, total_length=3.5, roadmap_size=100, iterations=50):
    """Erstellt Roboter mit n_dof in fixer Testumgebung, führt LazyPRM durch und visualisiert."""

    # Falls keine Hindernisse übergeben wurden → Standardhindernisse
    if obst is None:
        obst = dict()
        obst["obs1"] = LineString([(-2, 0), (-0.8, 0)]).buffer(0.5)
        obst["obs2"] = LineString([(2, 0), (2, 1)]).buffer(0.2)
        obst["obs3"] = LineString([(-1, 2), (1, 2)]).buffer(0.1)

    # Roboter erzeugen
    r = PlanarRobot(n_joints=n_dof, total_length=total_length)
    limits = [[-3.14, 3.14]] * r.dim
    environment = KinChainCollisionChecker(r, obst, limits=limits, fk_resolution=.2)

    # Start- und Zielkonfigurationen
    start_joint_angles = generate_consistent_joint_config(r.dim, total_angle=2.0, curvature=0.45)
    end_joint_angles = generate_consistent_joint_config(r.dim, total_angle=-1.85, curvature=-0.45)

    start = [np.array(start_joint_angles, dtype=np.float32)]
    goal = [np.array(end_joint_angles, dtype=np.float32)]

    # Zeichnen
    fig_local = plt.figure(figsize=(7, 7))
    ax = fig_local.add_subplot(1, 1, 1)
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    environment.drawObstacles(ax, True)
    r.move(sp.Matrix(start_joint_angles))
    planarRobotVisualize(r, ax)
    r.move(sp.Matrix(end_joint_angles))
    planarRobotVisualize(r, ax)

    return r, environment, start, goal, fig_local, ax