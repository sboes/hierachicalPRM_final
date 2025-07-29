# planners/lazy_prm.py

import numpy as np
import networkx as nx
import random
from scipy.spatial import cKDTree
from utils.collision_checker import CollisionChecker

class LazyPRM:
    def __init__(self, collision_checker, limits=((0.0, 22.0), (0.0, 22.0))):
        self._collision_checker = collision_checker
        self.scene = collision_checker.scene  # Damit die Visualisierung funktioniert
        self.limits = limits
        self.graph = nx.Graph()
        self.last_node_id = 0
        self.colliding_edges = []
        self.non_colliding_edges = []

    def _point_in_collision(self, pos):
        return self._collision_checker.point_in_collision(pos)

    def _line_in_collision(self, p1, p2):
        return self._collision_checker.line_in_collision(p1, p2)

    def _get_random_free_position(self, max_tries=1000):
        for _ in range(max_tries):
            x = random.uniform(*self.limits[0])
            y = random.uniform(*self.limits[1])
            pos = np.array([x, y])
            if not self._point_in_collision(pos):
                return pos
        raise RuntimeError("No free sample found.")

    def _build_roadmap(self, num_nodes, k_nearest):
        added_ids = []
        for _ in range(num_nodes):
            pos = self._get_random_free_position()
            node_id = self.last_node_id
            self.graph.add_node(node_id, pos=pos)
            added_ids.append(node_id)
            self.last_node_id += 1

        pos_dict = nx.get_node_attributes(self.graph, 'pos')
        node_ids = list(pos_dict.keys())
        positions = np.array([pos_dict[n] for n in node_ids])
        if len(positions) < 2:
            return

        tree = cKDTree(positions)
        for i, node_id in enumerate(node_ids):
            pos = positions[i]
            dists, idxs = tree.query(pos, k=min(k_nearest + 1, len(positions)))
            for j in idxs:
                neighbor_id = node_ids[j]
                if neighbor_id == node_id:
                    continue
                if not self.graph.has_edge(node_id, neighbor_id):
                    self.graph.add_edge(node_id, neighbor_id)

    def _connect_to_graph(self, label, pos, k=5):
        self.graph.add_node(label, pos=pos)
        pos_dict = nx.get_node_attributes(self.graph, 'pos')
        keys = list(pos_dict.keys())
        positions = np.array([pos_dict[k] for k in keys])
        if len(positions) == 0:
            return False

        tree = cKDTree(positions)
        dists, idxs = tree.query(pos, k=min(k, len(positions)))
        if not isinstance(idxs, list) and not isinstance(idxs, np.ndarray):
            idxs = [idxs]

        for j in idxs:
            nid = keys[j]
            if nid == label:
                continue
            if not self._line_in_collision(pos, pos_dict[nid]):
                self.graph.add_edge(label, nid)
                return True
        return False

    def _check_path_lazy(self, path):
        for u, v in zip(path[:-1], path[1:]):
            p1 = self.graph.nodes[u]["pos"]
            p2 = self.graph.nodes[v]["pos"]
            if self._line_in_collision(p1, p2):
                self.graph.remove_edge(u, v)
                self.colliding_edges.append((u, v))
                return False
            else:
                self.non_colliding_edges.append((u, v))
        return True

    def plan_path(self, start_list, goal_list, config):
        self.graph.clear()
        self.colliding_edges = []
        self.non_colliding_edges = []
        self.last_node_id = 0

        start = np.array(start_list[0])
        goal = np.array(goal_list[0])

        if self._point_in_collision(start) or self._point_in_collision(goal):
            return []

        self.graph.add_node("start", pos=start)
        self.graph.add_node("goal", pos=goal)

        self._build_roadmap(config["initialRoadmapSize"], config["kNearest"])
        self._connect_to_graph("start", start, k=config["kNearest"])
        self._connect_to_graph("goal", goal, k=config["kNearest"])

        for _ in range(config["maxIterations"]):
            try:
                path = nx.shortest_path(self.graph, "start", "goal")
            except nx.NetworkXNoPath:
                self._build_roadmap(config["updateRoadmapSize"], config["kNearest"])
                continue

            if self._check_path_lazy(path):
                return path

        return []

    def get_path_coordinates(self, path):
        return [self.graph.nodes[n]["pos"] for n in path]
