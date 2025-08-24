# coding: utf-8
"""
Hierarchical planner built on top of *lectures* modules (if available).
Replicates the visualization style of the 0.1 / 0.2 / 0.3 demo notebooks
(Basic PRM, Lazy PRM, Visibility PRM) by delegating to the provided utils.

Structure
---------
- LectureCollisionCheckerAdapter : wraps (scene, limits) to lectures' PlanerBase API
- make_basic_prm / make_lazy_prm / make_vis_prm : constructors using lectures
- HierarchicalPlanner : uses VisPRM as high-level, PRM/LazyPRM as local planner
- Visualization helpers: thin wrappers that call utils visualizers if installed
"""

from __future__ import annotations
import os, sys, math, time, random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Sequence, Optional

import numpy as np

# --- adaptive sys.path for project layout ---
for cand in (".","./lectures","./planners","./utils","./utils/utils"):
    if os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)

# --- try to import lectures ---
try:
    from lectures.IPBasicPRM import BasicPRM as LectBasicPRM
except Exception:
    LectBasicPRM = None

try:
    from lectures.IPLazyPRM import LazyPRM as LectLazyPRM
except Exception:
    LectLazyPRM = None

try:
    from lectures.IPVisibilityPRM import VisPRM as LectVisPRM
except Exception:
    LectVisPRM = None

# --- utils visualizers (VL look) ---
_vis_basic = None
_vis_lazy = None
_vis_vis = None
try:
    from basic_prm_visualize import basic_prm_visualize as _vis_basic
except Exception:
    try:
        from utils.basic_prm_visualize import basic_prm_visualize as _vis_basic
    except Exception:
        pass
try:
    from lazy_prm_visualize import lazy_prm_visualize as _vis_lazy
except Exception:
    try:
        from utils.lazy_prm_visualize import lazy_prm_visualize as _vis_lazy
    except Exception:
        pass
try:
    from visibility_prm_visualize import visibility_prm_visualize as _vis_vis
except Exception:
    try:
        from utils.visibility_prm_visualize import visibility_prm_visualize as _vis_vis
    except Exception:
        pass


# ---------------------------------------------------------------------
# Collision checker adapter (scene, limits) -> lectures PlanerBase API
# ---------------------------------------------------------------------

from shapely.geometry import Point, LineString, Polygon

@dataclass
class LectureCollisionCheckerAdapter:
    scene: Dict[str, Polygon]
    limits: Tuple[Tuple[float,float], Tuple[float,float]]  # ((xmin,xmax),(ymin,ymax))

    # --- API expected by lectures.PlanerBase / IPPRMBase ---
    def getDim(self) -> int:
        return 2
    def getEnvironmentLimits(self):
        return self.limits
    def pointInCollision(self, pos: Sequence[float]) -> bool:
        pt = Point(float(pos[0]), float(pos[1]))
        for obs in self.scene.values():
            if obs.intersects(pt) or obs.touches(pt):
                return True
        return False
    def lineInCollision(self, a: Sequence[float], b: Sequence[float]) -> bool:
        seg = LineString([(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))])
        for obs in self.scene.values():
            if obs.crosses(seg) or obs.intersects(seg) or obs.touches(seg):
                return True
        return False


# -----------------------------
# Planner constructors (lectures)
# -----------------------------

def make_basic_prm(cc: LectureCollisionCheckerAdapter,
                   radius: float = 3.0, num_nodes: int = 500, use_kdtree: bool = True):
    if LectBasicPRM is None:
        raise ImportError("lectures.IPBasicPRM not found")
    planner = LectBasicPRM(cc)
    planner.scene = cc.scene   # for visualizers
    planner.limits = cc.limits
    # config template to reuse later
    cfg = {"radius": radius, "numNodes": num_nodes, "useKDTree": use_kdtree}
    return planner, cfg

def make_lazy_prm(cc: LectureCollisionCheckerAdapter,
                  initial: int = 200, k_nearest: int = 15, update: int = 80, max_iter: int = 20):
    if LectLazyPRM is None:
        raise ImportError("lectures.IPLazyPRM not found")
    planner = LectLazyPRM(cc)
    planner.scene = cc.scene
    planner.limits = cc.limits
    cfg = {"initialRoadmapSize": initial, "kNearest": k_nearest,
           "updateRoadmapSize": update, "maxIterations": max_iter}
    return planner, cfg

def make_vis_prm(cc: LectureCollisionCheckerAdapter, ntry: int = 5000):
    if LectVisPRM is None:
        raise ImportError("lectures.IPVisibilityPRM not found")
    planner = LectVisPRM(cc)
    planner.scene = cc.scene
    planner.limits = cc.limits
    cfg = {"ntry": ntry}
    return planner, cfg


# ----------------------------------
# Visualization helpers (VL look)
# ----------------------------------

def visualize_basic(planner, path_nodes=None, ax=None):
    if _vis_basic is None:
        raise RuntimeError("basic_prm_visualize not found in utils")
    sol = path_nodes or []
    return _vis_basic(planner, sol, ax=ax)

def visualize_lazy(planner, path_nodes=None, ax=None):
    if _vis_lazy is None:
        raise RuntimeError("lazy_prm_visualize not found in utils")
    sol = path_nodes or []
    return _vis_lazy(planner, sol, ax=ax)

def visualize_visibility(planner, path_nodes=None, ax=None):
    if _vis_vis is None:
        raise RuntimeError("visibility_prm_visualize not found in utils")
    sol = path_nodes or []
    return _vis_vis(planner, sol, ax=ax)


# ----------------------------------
# Hierarchical Planner (lectures core)
# ----------------------------------

class HierarchicalPlanner:
    """
    High-level: VisPRM (guard roadmap from lectures)
    Low-level:  BasicPRM or LazyPRM (per segment)
    """
    def __init__(self, scene: Dict[str, Polygon], limits: Tuple[Tuple[float,float], Tuple[float,float]],
                 internal: str = "prm",
                 vis_ntry: int = 8000,
                 basic_radius: float = 3.0, basic_nodes: int = 600,
                 lazy_initial: int = 250, lazy_k: int = 18, lazy_update: int = 120, lazy_max_iter: int = 30):
        self.scene = scene
        self.limits = limits
        self.cc = LectureCollisionCheckerAdapter(scene, limits)

        self.internal = internal.lower()
        self.vis_ntry = int(vis_ntry)
        self.basic_radius = float(basic_radius)
        self.basic_nodes = int(basic_nodes)
        self.lazy_initial = int(lazy_initial)
        self.lazy_k = int(lazy_k)
        self.lazy_update = int(lazy_update)
        self.lazy_max_iter = int(lazy_max_iter)

        # will be created on plan()
        self.global_planner = None
        self.global_cfg = None
        self.segment_solutions: List[List[Tuple[float,float]]] = []
        self.segment_planners: List[object] = []  # keep for visualization

    def _make_local(self):
        if self.internal == "lazy":
            return make_lazy_prm(self.cc, self.lazy_initial, self.lazy_k, self.lazy_update, self.lazy_max_iter)
        else:
            return make_basic_prm(self.cc, self.basic_radius, self.basic_nodes, True)

    def plan(self, start: Tuple[float,float], goal: Tuple[float,float]):
        # build high-level roadmap
        vis, cfg = make_vis_prm(self.cc, self.vis_ntry)
        self.global_planner, self.global_cfg = vis, cfg

        startList = [list(start)]
        goalList = [list(goal)]
        path_nodes = vis.planPath(startList, goalList, cfg)  # list of node ids including "start"/"goal"
        if not path_nodes:
            self.segment_solutions = []
            self.segment_planners = []
            return []

        # Extract actual 2D positions of the guard path (including start/goal positions)
        pos_attr = vis.graph.nodes
        waypoint_xy: List[Tuple[float,float]] = [tuple(pos_attr[n]['pos']) for n in path_nodes]

        # for visualization symmetry, save global solution nodes
        self.global_solution_nodes = path_nodes

        # stitch local plans for each consecutive pair of waypoints
        full_xy: List[Tuple[float,float]] = [waypoint_xy[0]]
        self.segment_solutions = []
        self.segment_planners = []
        for a, b in zip(waypoint_xy[:-1], waypoint_xy[1:]):
            local, lcfg = self._make_local()
            # call lecture planner
            pn = local.planPath([list(a)], [list(b)], lcfg)  # nodes list; local.graph has 'pos'
            if not pn:
                self.segment_solutions = []
                self.segment_planners = []
                return []
            # convert nodes to xy
            posA = local.graph.nodes
            seg_xy = [tuple(posA[n]['pos']) for n in pn]
            # accumulate
            self.segment_planners.append(local)
            self.segment_solutions.append(seg_xy)
            # append, skipping first (duplicate)
            full_xy += seg_xy[1:]

        return full_xy


# ----------------------------------
# 2D animation (VL-like)
# ----------------------------------

import matplotlib.pyplot as plt
from matplotlib import animation

def animate_path_vl_style(scene, limits, path_xy: List[Tuple[float,float]], title="Path Animation", interval=80):
    """
    Simple dot animation in the same look&feel:
    - obstacles as filled patches
    - start/goal markers
    - path drawn progressively
    """
    fig, ax = plt.subplots(figsize=(8,8))
    # draw obstacles
    for obs in scene.values():
        if hasattr(obs,'exterior'):
            xs, ys = obs.exterior.xy
            ax.fill(xs, ys, color='lightcoral', alpha=0.6)
        else:
            xs, ys = obs.xy
            ax.fill(xs, ys, color='lightcoral', alpha=0.6)

    # draw limits & grid
    ax.set_xlim(limits[0]); ax.set_ylim(limits[1]); ax.set_aspect('equal'); ax.grid(True)
    ax.set_title(title)

    # precompute arrays
    xs = [p[0] for p in path_xy]
    ys = [p[1] for p in path_xy]

    # draw static full path (thin), then animate bold segment + dot
    ax.plot(xs, ys, linewidth=1.0, alpha=0.5)

    (line,) = ax.plot([], [], linewidth=3.0, color='g')
    (dot,)  = ax.plot([], [], 'o', color='g')

    def init():
        line.set_data([], [])
        dot.set_data([], [])
        return line, dot

    def update(i):
        if i < 1:
            line.set_data([xs[0]],[ys[0]])
            dot.set_data(xs[0], ys[0])
        else:
            line.set_data(xs[:i+1], ys[:i+1])
            dot.set_data(xs[i], ys[i])
        return line, dot

    frames = max(2, len(xs))
    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=interval)
    return ani, fig, ax
