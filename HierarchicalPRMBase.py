# coding: utf-8
"""
Hierarchical PRM Base Class
"""
from lectures.IPPRMBase import PRMBase
import networkx as nx
import numpy as np
from scipy.spatial import distance
from lectures.IPVisibilityPRM import VisibilityStatsHandler
from lectures.IPPerfMonitor import IPPerfMonitor
from scipy.spatial import cKDTree
from collections import defaultdict


class HierarchicalPRMBase(PRMBase):
    def __init__(self, _collChecker, _internalPlanner):
        super(HierarchicalPRMBase, self).__init__(_collChecker)
        self.internalPlanner = _internalPlanner
        self.graph = nx.Graph()
        self.statsHandler = VisibilityStatsHandler()
        self.internal_planner_stats = []
        self.attempt_stats = defaultdict(int)
        self.debug = False

    def set_debug(self, debug_mode):
        self.debug = debug_mode
        if hasattr(self.internalPlanner, 'set_debug'):
            self.internalPlanner.set_debug(debug_mode)

    def _configureInternalPlanner(self, pos1, pos2):
        """Configure the internal planner's search space"""
        padding = max(1.0, distance.euclidean(pos1, pos2) * 0.5)
        min_coords = np.minimum(pos1, pos2) - padding
        max_coords = np.maximum(pos1, pos2) + padding

        env_limits = self._collisionChecker.getEnvironmentLimits()
        for i in range(len(min_coords)):
            min_coords[i] = max(min_coords[i], env_limits[i][0])
            max_coords[i] = min(max_coords[i], env_limits[i][1])

        if hasattr(self.internalPlanner, 'setSamplingBounds'):
            bounds = list(zip(min_coords, max_coords))
            self.internalPlanner.setSamplingBounds(bounds)

    def _getInternalPlannerConfig(self):
        return {
            "radius": 1.0,
            "numNodes": 20,
            "useKDTree": True,
            "initialRoadmapSize": 15,
            "updateRoadmapSize": 5,
            "kNearest": 5,
            "maxIterations": 10
        }

    def _log(self, message):
        if self.debug:
            print(message)

    def _isVisible(self, pos1, pos2):
        """Use internal planner to check visibility with detailed logging"""
        self._log(f"\n=== Testing visibility between {pos1} and {pos2} ===")
        self._configureInternalPlanner(pos1, pos2)
        config = self._getInternalPlannerConfig()

        self.internalPlanner.graph.clear()
        path = self.internalPlanner.planPath([pos1], [pos2], config)

        result = len(path) > 0
        self._log(f"Visibility check {'succeeded' if result else 'failed'}")

        self.internal_planner_stats.append({
            'graph': self.internalPlanner.graph.copy(),
            'start': pos1,
            'goal': pos2,
            'success': result,
            'nodes': len(self.internalPlanner.graph.nodes()),
            'edges': len(self.internalPlanner.graph.edges())
        })

        self.attempt_stats['visibility_checks'] += 1
        if result:
            self.attempt_stats['successful_visibility'] += 1

        return result

    @IPPerfMonitor
    def planPath(self, startList, goalList, config, debug=False):
        """Main planning method with enhanced feedback"""
        self.set_debug(debug)
        self.graph.clear()
        self.internal_planner_stats = []
        self.attempt_stats = defaultdict(int)

        checkedStartList, checkedGoalList = self._checkStartGoal(startList, goalList)
        self._log(f"\n=== Starting planning from {checkedStartList[0]} to {checkedGoalList[0]} ===")

        # Learn roadmap
        self._learnRoadmap(config.get("ntry", 50))

        # Analyze roadmap
        self._log("\n=== Roadmap Analysis ===")
        self._log(f"Total nodes: {len(self.graph.nodes())}")
        self._log(f"Total edges: {len(self.graph.edges())}")
        self._log(f"Connected components: {nx.number_connected_components(self.graph)}")

        # Try to connect start and goal
        start_connected, goal_connected = False, False
        start_connection_info = self._connectNode("start", checkedStartList[0])
        goal_connection_info = self._connectNode("goal", checkedGoalList[0])

        # Diagnostic output
        if not start_connection_info['success']:
            self._log("\n❌ Start connection failed because:")
            self._log(f"- Nearest node: {start_connection_info['nearest_node']}")
            self._log(f"- Distance: {start_connection_info['distance']:.2f}")
            self._log(f"- Collision: {start_connection_info['in_collision']}")

        if not goal_connection_info['success']:
            self._log("\n❌ Goal connection failed because:")
            self._log(f"- Nearest node: {goal_connection_info['nearest_node']}")
            self._log(f"- Distance: {goal_connection_info['distance']:.2f}")
            self._log(f"- Collision: {goal_connection_info['in_collision']}")

        # Try to find path
        path = []
        if start_connection_info['success'] and goal_connection_info['success']:
            try:
                path = nx.shortest_path(self.graph, "start", "goal")
                self._log(f"\n✅ Found path with {len(path)} nodes")
            except nx.NetworkXNoPath:
                self._log("\n❌ No path between start and goal (roadmap disconnected)")

        # Generate final report
        self._generate_final_report(start_connection_info, goal_connection_info, bool(path))

        # Return both path and diagnostic info
        return {
            'path': path,
            'roadmap': self.graph,
            'stats': dict(self.attempt_stats),
            'start_connection': start_connection_info,
            'goal_connection': goal_connection_info,
            'internal_planner_stats': self.internal_planner_stats
        }

    def _connectNode(self, node_name, node_pos):
        """Attempt to connect a node (start/goal) to roadmap"""
        posList = nx.get_node_attributes(self.graph, 'pos')
        if not posList:
            return {'success': False, 'reason': 'no_nodes_in_roadmap'}

        kdTree = cKDTree(list(posList.values()))
        result = kdTree.query(node_pos, k=min(5, len(posList)))

        connection_info = {
            'success': False,
            'attempts': [],
            'nearest_node': None,
            'distance': float('inf'),
            'in_collision': False
        }

        for idx in np.atleast_1d(result[1]):
            if idx >= len(posList):
                continue

            node_id = list(posList.keys())[idx]
            candidate_pos = self.graph.nodes[node_id]['pos']
            dist = distance.euclidean(node_pos, candidate_pos)
            collision = self._collisionChecker.lineInCollision(node_pos, candidate_pos)

            attempt = {
                'node_id': node_id,
                'position': candidate_pos,
                'distance': dist,
                'collision': collision
            }
            connection_info['attempts'].append(attempt)

            if not collision:
                self.graph.add_node(node_name, pos=node_pos, color='lightgreen', nodeType='Terminal')
                self.graph.add_edge(node_name, node_id)
                connection_info.update({
                    'success': True,
                    'connected_to': node_id,
                    'nearest_node': candidate_pos,
                    'distance': dist
                })
                break
            else:
                if dist < connection_info['distance']:
                    connection_info.update({
                        'nearest_node': candidate_pos,
                        'distance': dist,
                        'in_collision': True
                    })

        self.attempt_stats[f"{node_name}_connection_attempts"] = len(connection_info['attempts'])
        if connection_info['success']:
            self.attempt_stats[f"{node_name}_connection_success"] = 1
        else:
            # Add failed terminal node for visualization
            self.graph.add_node(node_name, pos=node_pos, color='#ff9999', nodeType='FailedTerminal')

        return connection_info

    def _generate_final_report(self, start_info, goal_info, path_found):
        """Generate comprehensive planning report"""
        self._log("\n=== Planning Statistics ===")
        self._log(f"Visibility checks: {self.attempt_stats['visibility_checks']}")
        self._log(f"Successful visibility: {self.attempt_stats['successful_visibility']}")
        self._log(f"Guard nodes: {len([n for n, d in self.graph.nodes(data=True) if d.get('nodeType') == 'Guard'])}")
        self._log(
            f"Connection nodes: {len([n for n, d in self.graph.nodes(data=True) if d.get('nodeType') == 'Connection'])}")

        if not path_found:
            self._log("\n=== Failure Analysis ===")
            if len(self.graph.nodes()) == 0:
                self._log("- No nodes in roadmap (sampling failed)")
            elif not start_info['success'] and not goal_info['success']:
                self._log("- Both start and goal failed to connect")
                if start_info['in_collision'] and goal_info['in_collision']:
                    self._log("  - Both failed due to collision")
                elif start_info['distance'] > 5 or goal_info['distance'] > 5:
                    self._log("  - Potential issue: Sampling too sparse")
            elif not start_info['success']:
                self._log("- Start failed to connect")
            elif not goal_info['success']:
                self._log("- Goal failed to connect")
            else:
                self._log("- Roadmap is disconnected (no path between components)")

    @IPPerfMonitor
    def _learnRoadmap(self, ntry):
        """Learn the roadmap with detailed logging"""
        nodeNumber = 0
        currTry = 0

        while currTry < ntry:
            q_pos = self._getRandomFreePosition()
            self._log(f"\nSample #{nodeNumber}: {q_pos}")

            if self.statsHandler:
                self.statsHandler.addNodeAtPos(nodeNumber, q_pos)

            g_vis = None
            merged = False

            for comp in nx.connected_components(self.graph):
                found = False
                merged = False

                for g in comp:
                    if self.graph.nodes()[g]['nodeType'] == 'Guard':
                        self._log(f"Testing connection to guard node {g} at {self.graph.nodes()[g]['pos']}")

                        if self.statsHandler:
                            self.statsHandler.addVisTest(nodeNumber, g)

                        if self._isVisible(q_pos, self.graph.nodes()[g]['pos']):
                            found = True
                            if g_vis == None:
                                g_vis = g
                                self._log(f"First visible guard: {g}")
                            else:
                                self.graph.add_node(nodeNumber, pos=q_pos,
                                                    color='lightblue', nodeType='Connection')
                                self.graph.add_edge(nodeNumber, g)
                                self.graph.add_edge(nodeNumber, g_vis)
                                merged = True
                                self._log(f"Connected to guards {g} and {g_vis}")

                        if found:
                            break

                if merged:
                    break

            if (merged == False) and (g_vis == None):
                self.graph.add_node(nodeNumber, pos=q_pos, color='red', nodeType='Guard')
                self._log("New guard node created")
                currTry = 0
            else:
                currTry += 1
                self._log(f"Current try count: {currTry}/{ntry}")

            nodeNumber += 1
            self._log(f"Roadmap status: {len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges")