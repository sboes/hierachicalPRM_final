# utils/collision_checker.py

import numpy as np
from shapely.geometry import Point, LineString

class CollisionChecker:
    def __init__(self, scene, limits=((0, 22), (0, 22))):
        self.scene = scene
        self.limits = limits

    def drawObstacles(self, ax):
        for obs in self.scene.values():
            if hasattr(obs, 'exterior'):
                xs, ys = obs.exterior.xy
                ax.fill(xs, ys, color='lightcoral', alpha=0.6)
            else:
                # Für den Fall, dass das Hindernis keine exterior-Eigenschaft hat
                xs, ys = obs.xy
                ax.fill(xs, ys, color='lightcoral', alpha=0.6)

    def point_in_collision(self, pos):
        pt = Point(pos[0], pos[1])
        return any(obstacle.intersects(pt) for obstacle in self.scene.values())

    def line_in_collision(self, p1, p2):
        line = LineString([p1, p2])
        return any(obstacle.intersects(line) for obstacle in self.scene.values())
