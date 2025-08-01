# coding: utf-8
"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein). It is based on the slides/notebooks given during the course, so please **read this information first**

Important remark: Kudos to Gergely Soti, who did provide this code to integrate planar robots into the lecture

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import sympy as sp
import numpy as np


class PlanarJoint:
    def __init__(self, a=1.5, init_theta=0, id=0):
        self.a = a
        self.theta = init_theta
        self.sym_a, self.sym_theta = sp.symbols(f'a_{id} theta_{id}')
        self.rot = sp.Matrix([[sp.cos(self.sym_theta), -sp.sin(self.sym_theta)],
                              [sp.sin(self.sym_theta), sp.cos(self.sym_theta)]])
        self.trans = sp.Matrix([self.sym_a * sp.cos(self.sym_theta),
                                self.sym_a * sp.sin(self.sym_theta)])
        last_row = sp.Matrix([[0, 0, 1]])
        self.M = sp.Matrix.vstack(sp.Matrix.hstack(self.rot, self.trans), last_row)

    def get_subs(self):
        return {self.sym_a: self.a, self.sym_theta: self.theta}

    def get_transform(self):
        return np.squeeze(np.array(self.M.subs(self.get_subs()) * sp.Matrix([0, 0, 1]))).astype(np.float32)[:2]

    def move(self, new_theta):
        self.theta = new_theta


class PlanarRobot:
    def __init__(self, n_joints=2, total_length=3.0):
        self.dim = n_joints
        a_per_joint = total_length / n_joints
        self.joints = [PlanarJoint(a=a_per_joint, id=i) for i in range(self.dim)]
        self.Ms = [sp.eye(3)]
        for joint in self.joints:
            self.Ms.append(self.Ms[-1] * joint.M)

    def get_transforms(self):
        ts = []
        sub = {}
        for joint in self.joints:
            sub = {**sub, **joint.get_subs()}
        for M in self.Ms:
            ts.append(np.squeeze(np.array(M.subs(sub) * sp.Matrix([0, 0, 1]))).astype(np.float32)[:2])
        return ts

    def move(self, new_thetas):
        assert len(new_thetas) == len(self.joints)
        for i in range(len(self.joints)):
            self.joints[i].move(new_thetas[i])


def generate_consistent_joint_config(dof, total_angle=1.85, curvature=0.45):
    """
    Generates a joint configuration that results in consistent shape
    regardless of number of joints.

    Args:
        dof (int): number of joints
        total_angle (float): the main direction angle (like the first joint)
        curvature (float): the remaining angle to be spread among other joints

    Returns:
        list of joint angles
    """
    if dof < 2:
        return [total_angle]

    # First joint gets the "main direction" angle
    first = total_angle

    # Remaining joints split the curvature evenly
    rest = [curvature / (dof - 1)] * (dof - 1)

    return [first] + rest


if __name__ == '__main__':
    j = PlanarJoint()
    print(j.get_transform())
    j.move(0.8)
    print(j.get_transform())
    j.move(0.6)
    print(j.get_transform())
    j.move(1.552)
    print(j.get_transform())

    r = PlanarRobot(n_joints=3, total_length=4.5)
    r.move([0.5, 0.7, 1.2])
    print(r.get_transforms())
