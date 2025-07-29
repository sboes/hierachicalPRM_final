# utils/planarRobot.py

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

class PlanarJoint:
    def __init__(self, a=1.5, init_theta=0, id=0):
        self.a = a
        self.theta = init_theta
        self.sym_a, self.sym_theta = sp.symbols(f'a_{id} theta_{id}')
        self.M = sp.Matrix([
            [sp.cos(self.sym_theta), -sp.sin(self.sym_theta), self.sym_a * sp.cos(self.sym_theta)],
            [sp.sin(self.sym_theta),  sp.cos(self.sym_theta), self.sym_a * sp.sin(self.sym_theta)],
            [0,                      0,                       1]
        ])

    def get_subs(self):
        return {self.sym_a: self.a, self.sym_theta: self.theta}

    def move(self, theta):
        self.theta = theta

class PlanarRobot:
    def __init__(self, n_joints=3, link_lengths=None):
        self.dim = n_joints
        if link_lengths is None:
            link_lengths = [1.5] * n_joints
        self.joints = [PlanarJoint(link_lengths[i], id=i) for i in range(n_joints)]
        self.Ms = [sp.eye(3)]
        for joint in self.joints:
            self.Ms.append(self.Ms[-1] * joint.M)

    def move(self, thetas):
        for joint, theta in zip(self.joints, thetas):
            joint.move(theta)

    def get_transforms(self):
        subs = {}
        for joint in self.joints:
            subs.update(joint.get_subs())
        return [np.array((M.subs(subs) @ sp.Matrix([0, 0, 1]))[:2]).astype(np.float32).flatten() for M in self.Ms]

    def draw(self, ax, color='green'):
        points = self.get_transforms()
        for i in range(1, len(points)):
            x = [points[i - 1][0], points[i][0]]
            y = [points[i - 1][1], points[i][1]]
            ax.plot(x, y, color=color, linewidth=2)
        ax.plot(*points[-1], 'o', color='blue')

    def animate_motion(self, trajectory, ax, interval=200):
        from matplotlib.animation import FuncAnimation

        def update(i):
            ax.clear()
            self.move(trajectory[i])
            self.draw(ax)
            ax.set_xlim(-self.dim * 2, self.dim * 2)
            ax.set_ylim(-self.dim * 2, self.dim * 2)
            ax.set_title(f"Step {i}")
            ax.set_aspect("equal")

        ani = FuncAnimation(plt.gcf(), update, frames=len(trajectory), interval=interval)
        plt.show()
