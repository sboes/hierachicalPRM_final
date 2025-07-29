import numpy as np
from shapely.geometry import LineString
from utils.planarRobot import interpolate_line
from utils.collision_checker import CollisionChecker


class KinChainCollisionChecker(CollisionChecker):
    def __init__(self, kin_chain, scene, limits=[[-3.0, 3.0], [-3.0, 3.0]], statistic=None, fk_resolution=0.1):
        super(KinChainCollisionChecker, self).__init__(scene, limits)  # Nur scene und limits übergeben
        self.statistic = statistic  # Speichere statistic als Instanzvariable
        if len(limits) != 2 or len(limits[0]) != kin_chain.dim or len(limits[1]) != kin_chain.dim:
            raise ValueError("Limits must be a list [lower_limits, upper_limits] matching robot dimensions.")
        self.kin_chain = kin_chain
        self.fk_resolution = fk_resolution
        self.dim = self.kin_chain.dim

    def point_in_collision(self, pos):  # Geändert von pointInCollision
        self.kin_chain.move(pos)
        joint_positions = self.kin_chain.get_transforms()
        for i in range(1, len(joint_positions)):
            if self.segment_in_collision(joint_positions[i - 1], joint_positions[i]):
                return True
        return False

    def line_in_collision(self, startPos, endPos):  # Geändert von lineInCollision
        assert (len(startPos) == self.getDim())
        assert (len(endPos) == self.getDim())
        steps = interpolate_line(startPos, endPos, self.fk_resolution)
        for pos in steps:
            if self.point_in_collision(pos):
                return True
        return False

    def segment_in_collision(self, startPos, endPos):  # Geändert von segmentInCollision
        for key, value in self.scene.items():
            if value.intersects(LineString([(startPos[0], startPos[1]), (endPos[0], endPos[1])])):
                return True
        return False

    def getDim(self):
        return self.dim