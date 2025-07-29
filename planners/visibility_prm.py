import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString
from shapely import plotting


class CollisionChecker:
    def __init__(self, scene, limits=((0, 22), (0, 22))):
        self.scene = scene
        self.limits = limits

    def pointInCollision(self, pos):
        point = Point(pos[0], pos[1])
        return any(obstacle.intersects(point) for obstacle in self.scene.values())

    def lineInCollision(self, p1, p2):
        line = LineString([p1, p2])
        return any(obstacle.intersects(line) for obstacle in self.scene.values())

    def drawObstacles(self, ax):
        for obstacle in self.scene.values():
            if hasattr(obstacle, 'exterior'):
                xs, ys = obstacle.exterior.xy
                ax.fill(xs, ys, color='lightcoral', alpha=0.6)


class VisibilityStatsHandler:
    def __init__(self):
        self.graph = nx.Graph()

    def add_node_at_pos(self, node_id, pos):
        self.graph.add_node(node_id, pos=pos, color='yellow')

    def add_vis_test(self, fr, to):
        self.graph.add_edge(fr, to)


class VisibilityPRM:
    def __init__(self, scene, limits=((0, 22), (0, 22))):
        self._collision_checker = CollisionChecker(scene, limits)
        self.limits = limits
        self.graph = nx.Graph()
        self.stats_handler = VisibilityStatsHandler()
        self.scene = scene  # Added for visualization access

    def _get_random_free_position(self):
        while True:
            pos = np.array([
                np.random.uniform(self.limits[0][0], self.limits[0][1]),
                np.random.uniform(self.limits[1][0], self.limits[1][1])
            ])
            if not self._collision_checker.pointInCollision(pos):
                return pos

    def _line_in_collision(self, p1, p2):
        return self._collision_checker.lineInCollision(p1, p2)

    def _check_start_goal(self, start, goal):
        for pt in start + goal:
            if self._collision_checker.pointInCollision(pt):
                raise ValueError("Start or goal position in collision.")
        return start, goal

    def _is_visible(self, pos1, pos2):
        return not self._line_in_collision(pos1, pos2)

    def _learn_roadmap(self, ntry):
        node_id = 0
        attempts = 0

        while attempts < ntry:
            q_pos = self._get_random_free_position()
            self.stats_handler.add_node_at_pos(node_id, q_pos)

            visible_guards = []
            for comp in nx.connected_components(self.graph):
                for g in comp:
                    if self.graph.nodes[g].get('nodeType') == 'Guard':
                        if self._is_visible(q_pos, self.graph.nodes[g]['pos']):
                            self.stats_handler.add_vis_test(node_id, g)
                            visible_guards.append(g)

            if len(visible_guards) == 0:
                self.graph.add_node(node_id, pos=q_pos, color='red', nodeType='Guard')
                attempts = 0  # Reset counter to ensure enough guards are placed
            elif len(visible_guards) == 1:
                # No connection needed, point sees only one guard
                attempts += 1
            else:
                # Connect two guards through new connection node
                self.graph.add_node(node_id, pos=q_pos, color='lightblue', nodeType='Connection')
                self.graph.add_edge(node_id, visible_guards[0])
                self.graph.add_edge(node_id, visible_guards[1])
                attempts += 1

            node_id += 1

    def _connect_endpoint(self, label, pos, k=10):
        self.graph.add_node(label, pos=pos, color='green')

        pos_dict = nx.get_node_attributes(self.graph, 'pos')
        keys = list(pos_dict.keys())
        coords = np.array(list(pos_dict.values()))

        if len(coords) == 0:
            return False

        tree = cKDTree(coords)
        dists, idxs = tree.query(pos, k=min(k, len(coords)))

        connected = False
        for idx in np.atleast_1d(idxs):
            neighbor = keys[idx]
            if self._is_visible(pos, pos_dict[neighbor]):
                self.graph.add_edge(label, neighbor)
                connected = True

        return connected

    def plan_path(self, start_list, goal_list, config):
        self.graph.clear()

        start, goal = self._check_start_goal(start_list, goal_list)
        self._learn_roadmap(config.get("ntry", 100))

        success_start = self._connect_endpoint("start", start[0])
        success_goal = self._connect_endpoint("goal", goal[0])

        if not (success_start and success_goal):
            return []

        try:
            return nx.shortest_path(self.graph, "start", "goal")
        except nx.NetworkXNoPath:
            return []

    def get_path_coordinates(self, path):
        pos = nx.get_node_attributes(self.graph, 'pos')
        return [pos[p] for p in path]

    @property
    def stats(self):
        return self.stats_handler