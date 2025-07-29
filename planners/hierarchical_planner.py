# planners/hierarchical_planner.py

import numpy as np
import networkx as nx
from planners.visibility_prm import VisibilityPRM
from utils.collision_checker import CollisionChecker


class HierarchicalPlanner:
    def __init__(self, scene, limits, GlobalPlannerClass, LocalPlannerClass,
                 config_global, config_local):
        self.scene = scene
        self.limits = limits
        self.config_global = config_global
        self.config_local = config_local

        self.global_planner = GlobalPlannerClass(scene, limits)
        self.LocalPlannerClass = LocalPlannerClass

        self.global_path = []
        self.subpaths = []
        self.solution_path = []
        self.global_graph_snapshot = None
        self.expanded_graph = nx.Graph()
        self.subplanner_snapshots = []  # Für Visualisierung: speichert einzelne Subplaner

    def plan_path(self, start, goal):
        self.solution_path.clear()
        self.subpaths.clear()
        self.expanded_graph.clear()
        self.subplanner_snapshots.clear()

        self.global_path = self.global_planner.plan_path(start, goal, self.config_global)
        if not self.global_path:
            print("Kein Pfad im Sichtbarkeitsgraph gefunden.")
            return []

        self.global_graph_snapshot = self.global_planner.graph.copy()
        coords = self.global_planner.get_path_coordinates(self.global_path)
        print(f"Globaler Pfad: {self.global_path}")
        print(f"Koordinaten des globalen Pfads: {coords}")

        for i in range(len(coords) - 1):
            start_coords = coords[i]
            goal_coords = coords[i + 1]
            print(f"Segment {i}: {start_coords} → {goal_coords}")

            segment_vec = np.array(goal_coords) - np.array(start_coords)
            segment_length = np.linalg.norm(segment_vec)
            padding = min(max(segment_length, 0.5), 8.0)  # begrenzter, kontrollierter Puffer

            center = np.array(start_coords) + 0.5 * segment_vec
            min_x = max(center[0] - padding, self.limits[0][0])
            max_x = min(center[0] + padding, self.limits[0][1])
            min_y = max(center[1] - padding, self.limits[1][0])
            max_y = min(center[1] + padding, self.limits[1][1])
            local_limits = ((min_x, max_x), (min_y, max_y))

            print(f"→ Lokaler Planner für Segment {i}: {start_coords} → {goal_coords}")
            print(f"   Lokale Begrenzung (geclipped): {local_limits}")

            local_checker = CollisionChecker(self.scene, local_limits)
            local_planner = self.LocalPlannerClass(local_checker, limits=local_limits)

            sub_path = local_planner.plan_path([start_coords], [goal_coords], self.config_local)
            if not sub_path:
                print(f"❌ Keine lokale Verbindung zwischen {start_coords} und {goal_coords}.")
                return []

            coords_segment = local_planner.get_path_coordinates(sub_path)

            # Verwerfe Segment, wenn Koordinaten außerhalb globaler Limits sind
            if not all(
                    self.limits[0][0] <= p[0] <= self.limits[0][1] and
                    self.limits[1][0] <= p[1] <= self.limits[1][1]
                    for p in coords_segment
            ):
                print("❌ Knoten außerhalb globaler Grenzen – verwerfe Teilsegment.")
                return []

            self.subpaths.append(coords_segment)
            self.subplanner_snapshots.append((local_planner, sub_path))

            for j in range(len(coords_segment)):
                node_id = (i, j)
                self.expanded_graph.add_node(node_id, pos=coords_segment[j])
                if j > 0:
                    self.expanded_graph.add_edge((i, j - 1), (i, j))

        self.solution_path = [pt for segment in self.subpaths for pt in segment]
        return self.solution_path

    def get_global_graph(self):
        return self.global_graph_snapshot

    def get_expanded_graph(self):
        return self.expanded_graph

    def get_solution_path(self):
        return self.solution_path

    def get_visibility_path_nodes(self):
        return self.global_path

    def get_subplanners(self):
        return self.subplanner_snapshots
