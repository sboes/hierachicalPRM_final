
# coding: utf-8
"""
Hierarchical PRM (main: Visibility-PRM, internal: BasicPRM or LazyPRM)

This module composes existing lecture planners:
- lectures.IPVisibilityPRM.VisPRM  (main planner / "Hauptplaner")
- lectures.IPBasicPRM.BasicPRM     (internal planner option)
- lectures.IPLazyPRM.LazyPRM       (internal planner option)

Core idea (from the project task):
- In the main planner, the *line test* for connecting two configurations is
  replaced by a call to an internal path planner that tries to find a path
  between those two local configurations. If the internal planner finds a path,
  the connection in the main graph is accepted; otherwise it's rejected.
- The internal planner runs with a *limited search effort* (bounded roadmap
  size / radius / kNN etc.). You control this via the `inner_config` argument.

API is kept consistent with the lecture planners: `planPath(startList, goalList, config)`.
"""
from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import networkx as nx

# --- imports from lecture code ---
from lectures.IPVisibilityPRM import VisPRM
from lectures.IPBasicPRM import BasicPRM
from lectures.IPLazyPRM import LazyPRM
from lectures.IPPerfMonitor import IPPerfMonitor

# Optional: numpy only used for simple distance check
import numpy as np


class HierarchicalPRM(VisPRM):
    """
    Main planner: Visibility-PRM (superclass).
    Edge feasibility is delegated to an "internal" PRM (Basic or Lazy).

    Usage example:
        env = ...  # lectures.IPEnvironment.CollisionChecker
        planner = HierarchicalPRM(env, inner_type='basic')
        cfg_main = {'ntry': 80}  # visibility roadmap budget
        cfg_inner = {'type': 'basic', 'config': {'radius': 3.5, 'numNodes': 150}}
        path = planner.planPath([start], [goal], {'ntry': 80, 'inner': cfg_inner})
    """

    def __init__(self, collisionChecker, inner_type: str = 'basic'):
        super(HierarchicalPRM, self).__init__(collisionChecker)
        self._inner_type = inner_type.lower()
        self._edge_cache: Dict[Tuple[Tuple[float,float], Tuple[float,float]], bool] = {}

    # --- helper: build internal planner instance ---
    def _make_inner_planner(self, inner_type: str):
        t = inner_type.lower()
        if t == 'basic':
            return BasicPRM(self._collisionChecker)
        elif t == 'lazy':
            return LazyPRM(self._collisionChecker)
        else:
            raise ValueError(f"Unknown inner planner type: {inner_type}")

    # --- override visibility test so that it uses inner planner instead of line-of-sight ---
    def _isVisible(self, pos_a: List[float], pos_b: List[float], inner: Optional[Dict[str, Any]] = None) -> bool:
        """
        Return True if an internal PRM can find a (local) path between pos_a and pos_b
        under the limited inner configuration.
        A small cache avoids repeated solving of the same local query.
        """
        key = (tuple(np.round(pos_a, 5)), tuple(np.round(pos_b, 5)))
        if key in self._edge_cache:
            return self._edge_cache[key]

        # If no inner config is provided, fall back to straight visibility
        if inner is None:
            ok = not self._collisionChecker.lineInCollision(pos_a, pos_b)
            self._edge_cache[key] = ok
            return ok

        inner_type = inner.get('type', self._inner_type)
        inner_planner = self._make_inner_planner(inner_type)

        inner_cfg = inner.get('config', {})
        # sensible defaults if not provided
        if inner_type == 'basic':
            inner_cfg.setdefault('radius', 3.0)
            inner_cfg.setdefault('numNodes', 150)
        elif inner_type == 'lazy':
            inner_cfg.setdefault('numNodes', 150)
            inner_cfg.setdefault('kNN', 10)
            inner_cfg.setdefault('maxEdgeLen', 5.0)

        # Try to solve the local query quickly
        try:
            path = inner_planner.planPath([list(pos_a)], [list(pos_b)], inner_cfg)
            ok = len(path) > 0
        except Exception:
            ok = False

        self._edge_cache[key] = ok
        return ok

    @IPPerfMonitor
    def planPath(self, startList: List[List[float]], goalList: List[List[float]], config: Dict[str, Any]):
        """
        Args:
            startList: list with single start configuration
            goalList:  list with single goal configuration
            config:
                - 'ntry':       int, budget for learning the visibility roadmap
                - 'inner':      dict, configuration for the internal planner used
                                for local edge feasibility:
                                {
                                   'type': 'basic' | 'lazy',
                                   'config': {...}   # params forwarded to inner planner
                                }
        Notes:
            - This overrides VisPRM.planPath by injecting our own _isVisible call
              that receives the 'inner' configuration.
            - We still reuse VisPRM's learning / merging logic. Only the boolean
              visibility test differs.
        """
        # reset main graph and edge cache
        self.graph = nx.Graph()
        self._edge_cache.clear()

        # Validate and extract configs
        inner_cfg = config.get('inner', None)
        ntry = config.get('ntry', 60)

        # We copy-paste minimal parts of VisPRM.planPath to pass inner_cfg down
        # without rewriting lecture code. The original VisPRM.planPath calls
        # self._learnRoadmap(ntry) and then connects start/goal using
        # self._isVisible(...). We keep the same structure but route visibility
        # checks through our override with inner_cfg.

        # --- Learn the visibility roadmap (guards/connectors) ---
        # we call the inherited method that relies on self._isVisible; our override
        # handles the 'inner' configuration by storing it on the instance for the duration
        self._current_inner_cfg = inner_cfg
        self._learnRoadmap_with_inner(ntry)  # custom wrapper
        # --- Connect start/goal to the roadmap using the same inner-based visibility ---
        from scipy.spatial import cKDTree
        import numpy as np

        checkedStartList, checkedGoalList = self._checkStartGoal(startList, goalList)

        # connect start
        posList = list(nx.get_node_attributes(self.graph, 'pos').values())
        if len(posList) == 0:
            return []
        kdTree = cKDTree(posList)
        nearest = kdTree.query(checkedStartList[0], k=min(10, len(posList)))[1]
        # ensure iterable
        if np.isscalar(nearest):
            nearest = [nearest]
        nodes = list(self.graph.nodes())
        start_connected = False
        for idx in nearest:
            to_node = nodes[idx]
            if self._isVisible(checkedStartList[0], self.graph.nodes[to_node]['pos'], self._current_inner_cfg):
                self.graph.add_node("start", pos=checkedStartList[0], color='lightgreen')
                self.graph.add_edge("start", to_node)
                start_connected = True
                break

        # connect goal
        nearest = kdTree.query(checkedGoalList[0], k=min(10, len(posList)))[1]
        if np.isscalar(nearest):
            nearest = [nearest]
        goal_connected = False
        for idx in nearest:
            to_node = nodes[idx]
            if self._isVisible(checkedGoalList[0], self.graph.nodes[to_node]['pos'], self._current_inner_cfg):
                self.graph.add_node("goal", pos=checkedGoalList[0], color='lightgreen')
                self.graph.add_edge("goal", to_node)
                goal_connected = True
                break

        if not (start_connected and goal_connected):
            return []

        try:
            path = nx.shortest_path(self.graph, "start", "goal")
        except nx.NetworkXNoPath:
            return []
        finally:
            # cleanup
            if hasattr(self, '_current_inner_cfg'):
                delattr(self, '_current_inner_cfg')

        return path

    # ---- small wrapper around the inherited learning routine, so we can inject inner_cfg into _isVisible calls ----
    @IPPerfMonitor
    def _learnRoadmap_with_inner(self, ntry: int):
        """
        A lightly adapted copy of VisPRM._learnRoadmap that forwards the
        inner-config into _isVisible. We keep the lecture's behaviour otherwise.
        """
        from scipy.spatial import cKDTree
        import numpy as np

        nodeNumber = 0
        currTry = 0
        while currTry < ntry:
            q_pos = self._getRandomFreePosition()
            if self.statsHandler:
                self.statsHandler.addNodeAtPos(nodeNumber, q_pos)

            g_vis = None
            merged = False
            posList = list(nx.get_node_attributes(self.graph, 'pos').values())
            # 1) find nearest guard(s) (if any exist)
            if len(posList) > 0:
                kdTree = cKDTree(posList)
                nearest = kdTree.query(q_pos, k=min(10, len(posList)))[1]
                if np.isscalar(nearest):
                    nearest = [nearest]

                # 2) find first visible guard and connect as connector
                for idx in nearest:
                    nearestNode = list(self.graph.nodes())[idx]
                    if self._isVisible(q_pos, self.graph.nodes[nearestNode]['pos'], self._current_inner_cfg):
                        self.graph.add_node(nodeNumber, pos=q_pos, color='blue', nodeType='Connector')
                        self.graph.add_edge(nodeNumber, nearestNode)
                        g_vis = nearestNode
                        break

            # 3) try to connect to another visible guard and merge components
            if g_vis is not None:
                for idx in nearest:
                    nearestNode = list(self.graph.nodes())[idx]
                    if nearestNode == g_vis:
                        continue
                    if self._isVisible(q_pos, self.graph.nodes[nearestNode]['pos'], self._current_inner_cfg):
                        self.graph.add_edge(nodeNumber, nearestNode)
                        merged = True
                        break

            if (merged is False) and (g_vis is None):
                self.graph.add_node(nodeNumber, pos=q_pos, color='red', nodeType='Guard')
                currTry = 0
            else:
                currTry += 1

            nodeNumber += 1
