import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from lectures.IPPlanarManipulator import PlanarRobot, generate_consistent_joint_config
from lectures.IPEnvironmentKin import KinChainCollisionChecker, planarRobotVisualize, animateSolution
import IPLazyPRM
import IPVISLazyPRM


def setup_and_plan(n_dof: int, obst=None, total_length=3.5, roadmap_size=100, iterations=50):
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

    # Planen mit LazyPRM
    lazyPRM = IPLazyPRM.LazyPRM(environment)
    lazyConfig = dict(
        initialRoadmapSize=roadmap_size,
        updateRoadmapSize=roadmap_size // 3,
        kNearest=15,
        maxIterations=iterations,
    )

    solution = lazyPRM.planPath(start, goal, lazyConfig)
    print(f"Solution for {n_dof}-DoF robot:", solution)

    # Animation
    animateSolution(lazyPRM, environment, solution, IPVISLazyPRM.lazyPRMVisualize, workSpaceLimits=[[-3, 3], [-3, 3]])

    return solution
