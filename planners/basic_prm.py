# planners/basic_prm.py

import numpy as np
import networkx as nx
from shapely.geometry import Point
from scipy.spatial.distance import euclidean
import random
from utils.collision_checker import CollisionChecker


# planners/basic_prm.py
class BasicPRM:
    def __init__(self, collision_checker, limits=((0.0, 22.0), (0.0, 22.0))):
        """
        Args:
            collision_checker: CollisionChecker instance
            limits ((xmin, xmax), (ymin, ymax)): bounds for random sampling
        """
        self._collision_checker = collision_checker
        self.scene = collision_checker.scene  # für Visualisierung
        self.limits = limits
        self.graph = nx.Graph()

    def _point_in_collision(self, pos):
        return self._collision_checker.point_in_collision(pos)

    def _line_in_collision(self, p1, p2):
        return self._collision_checker.line_in_collision(p1, p2)

    # ----- Sampling -----
    def _get_random_free_position(self, max_tries=1000):
        for _ in range(max_tries):
            x = random.uniform(*self.limits[0])
            y = random.uniform(*self.limits[1])
            pos = np.array([x, y])
            if not self._point_in_collision(pos):
                return pos
        raise RuntimeError("No collision-free sample found.")

    # ----- PRM Core -----
    def _nearest_neighbours(self, pos, radius):
        return [
            (node, data)
            for node, data in self.graph.nodes(data=True)
            if euclidean(data['pos'], pos) <= radius
        ]

    def _in_same_component(self, node1, node2):
        for component in nx.connected_components(self.graph):
            if node1 in component and node2 in component:
                return True
        return False

    def _build_roadmap(self, num_nodes, radius):
        node_id = 1
        while node_id <= num_nodes:
            pos = self._get_random_free_position()
            self.graph.add_node(node_id, pos=pos)

            neighbours = self._nearest_neighbours(pos, radius)
            for nid, ndata in neighbours:
                if self._in_same_component(node_id, nid):
                    continue
                if not self._line_in_collision(pos, ndata['pos']):
                    self.graph.add_edge(node_id, nid)

            node_id += 1

    def _connect_to_graph(self, label, pos, radius):
        self.graph.add_node(label, pos=pos)
        for nid, ndata in self._nearest_neighbours(pos, radius):
            if not self._line_in_collision(pos, ndata['pos']):
                self.graph.add_edge(label, nid)
                return True
        return False

    # ----- Path Planning -----
    def plan_path(self, start_list, goal_list, config):
        """
        Args:
            start_list: list of [x, y]
            goal_list: list of [x, y]
            config: dict with keys: numNodes, radius

        Returns:
            list of node IDs forming the path or []
        """
        self.graph.clear()
        start = np.array(start_list[0])
        goal = np.array(goal_list[0])

        if self._point_in_collision(start) or self._point_in_collision(goal):
            return []

        self._build_roadmap(config["numNodes"], config["radius"])

        if not self._connect_to_graph("start", start, config["radius"]):
            return []
        if not self._connect_to_graph("goal", goal, config["radius"]):
            return []

        try:
            return nx.shortest_path(self.graph, "start", "goal")
        except nx.NetworkXNoPath:
            return []

    def get_path_coordinates(self, path):
        return [self.graph.nodes[n]["pos"] for n in path]
