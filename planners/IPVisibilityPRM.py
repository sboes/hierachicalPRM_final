# coding: utf-8
"""
Visibility PRM mit lokal begrenztem Subplanner gemäß Projektaufgabe
  - Für jede Sichtbarkeitsprüfung wird ein interner PRM in einer kleinen Region (ROI)
    erzeugt und wieder verworfen.
  - Der ROI wird durch ein Rechteck um (pos, guardPos) plus Offset aus Config definiert.
  - Kanten im High-Level-Graph werden gewichtet nach euklidischer Distanz.
"""
from lectures.IPPRMBase import PRMBase
import networkx as nx
from scipy.spatial import cKDTree
from lectures.IPPerfMonitor import IPPerfMonitor
import numpy as np
import math
from lectures.IPLazyPRM import LazyPRM
from lectures.IPBasicPRM import BasicPRM

class VisPRM_ROI(PRMBase):
    def __init__(self, collChecker, statsHandler=None, prm_type='lazy'):
        super().__init__(collChecker)
        self.graph = nx.Graph()
        self.statsHandler = statsHandler
        self._LocalPRMClass = BasicPRM if prm_type=='basic' else LazyPRM
        self._visConfig = {}

    def _compute_roi(self, p1, p2):
        # ROI als Rechteck um p1/p2 plus margin
        m = self._visConfig.get('regionMargin', 1.0)
        xmin = min(p1[0], p2[0]) - m
        xmax = max(p1[0], p2[0]) + m
        ymin = min(p1[1], p2[1]) - m
        ymax = max(p1[1], p2[1]) + m
        return ([xmin, xmax], [ymin, ymax])

    def _isVisible(self, pos, guardPos):
        # build small ROI-checker
        roi_limits = self._compute_roi(pos, guardPos)
        # instantiate ROI collision checker of same type
        # assume self._collisionChecker has properties scene and can take new limits
        roi_checker = type(self._collisionChecker)(self._collisionChecker.scene, roi_limits)
        # build fresh sub-PRM in ROI
        prm_cfg = {
            'initialRoadmapSize': self._visConfig.get('initialRoadmapSize', 20),
            'updateRoadmapSize':   self._visConfig.get('updateRoadmapSize', 10),
            'kNearest':            self._visConfig.get('kNearest', 5),
            'maxIterations':       self._visConfig.get('maxIterations', 10),
            'radius':              self._visConfig.get('radius', 5.0),
            'numNodes':            self._visConfig.get('numNodes', 50)
        }
        subprm = self._LocalPRMClass(roi_checker)
        path = subprm.planPath([pos], [guardPos], prm_cfg)
        # Statistik
        if self.statsHandler:
            self.statsHandler.addVisTest(pos, guardPos)
        return bool(path)

    @IPPerfMonitor
    def _learnRoadmap(self, ntry):
        node_num = 0
        curr = 0
        while curr < ntry:
            q = self._getRandomFreePosition()
            if self.statsHandler:
                self.statsHandler.addNodeAtPos(node_num, q)

            best = None
            merged = False
            for comp in nx.connected_components(self.graph):
                for g in comp:
                    if self.graph.nodes[g].get('nodeType')!='Guard': continue
                    if self._isVisible(q, self.graph.nodes[g]['pos']):
                        if best is None:
                            best = g
                        else:
                            # connection node
                            self.graph.add_node(node_num, pos=q, color='lightblue', nodeType='Connection')
                            # gewichtete Kanten
                            for u in (g, best):
                                d = math.dist(q, self.graph.nodes[u]['pos'])
                                self.graph.add_edge(node_num, u, weight=d)
                            merged=True
                    if merged: break
                if merged: break

            if not merged and best is None:
                self.graph.add_node(node_num, pos=q, color='red', nodeType='Guard')
                curr=0
            else:
                curr+=1
            node_num+=1

    @IPPerfMonitor
    def planPath(self, startList, goalList, config):
        self._visConfig=config
        self.graph.clear()
        starts, goals = self._checkStartGoal(startList, goalList)
        self._learnRoadmap(config.get('ntry',40))
        posAttr = nx.get_node_attributes(self.graph,'pos')
        if not posAttr:
            return []
        kd = cKDTree(list(posAttr.values()))
        # attach start/goal
        for label, pts in [('start',starts),('goal',goals)]:
            p=pts[0]
            for idx in np.atleast_1d(kd.query(p,k=min(5,len(posAttr)))[1]):
                nid=list(posAttr.keys())[idx]
                if self._isVisible(p,posAttr[nid]):
                    d=math.dist(p,posAttr[nid])
                    self.graph.add_node(label,pos=p,color='lightgreen')
                    self.graph.add_edge(label,nid,weight=d)
                    break
        try:
            return nx.shortest_path(self.graph,'start','goal',weight='weight')
        except nx.NetworkXNoPath:
            return []

    def get_path_coordinates(self, high_path):
        coords=[]
        prm_cfg=self._visConfig
        for i in range(len(high_path)-1):
            u,v=high_path[i],high_path[i+1]
            p1=self.graph.nodes[u]['pos']; p2=self.graph.nodes[v]['pos']
            roi_limits=self._compute_roi(p1,p2)
            roi_checker=type(self._collisionChecker)(self._collisionChecker.scene, roi_limits)
            subprm=self._LocalPRMClass(roi_checker)
            sub= subprm.planPath([p1],[p2],prm_cfg)
            pts=[subprm.graph.nodes[n]['pos'] for n in sub]
            if i>0 and pts: pts=pts[1:]
            coords.extend(pts)
        return coords

    # coding: utf-8
    """
    Visibility PRM (Plan-based Connectivity)
    Hierarchischer Planner: Sichtbarkeitsprüfung ausschließlich über lokalen PRM (LazyPRM oder BasicPRM).
    Jeder Knoten wird als Guard betrachtet, und Verbindungen entstehen nur, wenn ein Sub-PRM einen Pfad findet.
    Keine direkten Linientests, keine Connection-Nodes, nur reine PRM-basierte Kanten.
    """
    from lectures.IPPRMBase import PRMBase
    import networkx as nx
    from scipy.spatial import cKDTree
    from lectures.IPPerfMonitor import IPPerfMonitor
    import numpy as np
    import math
    from lectures.IPLazyPRM import LazyPRM
    from lectures.IPBasicPRM import BasicPRM


class VisPRM_PBC(PRMBase):
    def __init__(self, collChecker, prm_type='lazy'):
        super().__init__(collChecker)
        self.graph = nx.Graph()
        # choose subplanner class
        self.SubPRMClass = BasicPRM if prm_type.lower() == 'basic' else LazyPRM
        self._visConfig = {}

    @IPPerfMonitor
    def _build_roadmap(self, num_samples):
        # Sample free configurations
        samples = [self._getRandomFreePosition() for _ in range(num_samples)]
        # add nodes
        for i, pos in enumerate(samples):
            self.graph.add_node(i, pos=pos)
        # Build neighbor structure
        pos_list = list(nx.get_node_attributes(self.graph, 'pos').values())
        kdt = cKDTree(pos_list)
        # For each node, try to connect to kNearest neighbors via local PRM
        k = self._visConfig.get('kNearest', 5)
        prm_args = {
            'initialRoadmapSize': self._visConfig.get('initialRoadmapSize', 20),
            'updateRoadmapSize': self._visConfig.get('updateRoadmapSize', 10),
            'kNearest': self._visConfig.get('kNearest', 5),
            'maxIterations': self._visConfig.get('maxIterations', 10),
            'radius': self._visConfig.get('radius', 5.0),
            'numNodes': self._visConfig.get('numNodes', 50)
        }
        for i, pos in enumerate(pos_list):
            # find neighbor indices (skip self)
            dists, idxs = kdt.query(pos, k=min(k + 1, len(pos_list)))
            for dist, j in zip(np.atleast_1d(dists)[1:], np.atleast_1d(idxs)[1:]):
                # only try each pair once
                if self.graph.has_edge(i, j):
                    continue
                # create fresh subplanner
                subprm = self.SubPRMClass(self._collisionChecker)
                subpath = subprm.planPath([pos], [pos_list[j]], prm_args)
                if subpath:
                    w = math.dist(pos, pos_list[j])
                    self.graph.add_edge(i, j, weight=w)

    @IPPerfMonitor
    def planPath(self, startList, goalList, config):
        # store config
        self._visConfig = config
        # clear any existing graph
        self.graph.clear()
        # collision-free start/goal
        starts, goals = self._checkStartGoal(startList, goalList)
        if not starts or not goals:
            return []
        # build roadmap
        self._build_roadmap(config.get('numSamples', 50))
        # attach start and goal as special nodes
        nodes_pos = nx.get_node_attributes(self.graph, 'pos')
        all_pos = list(nodes_pos.values())
        kdt = cKDTree(all_pos)
        prm_args = config
        for label, p in [('start', starts[0]), ('goal', goals[0])]:
            self.graph.add_node(label, pos=p)
            # connect to k nearest samples
            dists, idxs = kdt.query(p, k=min(config.get('kNearest', 5), len(all_pos)))
            for j in np.atleast_1d(idxs):
                idx_list = list(nodes_pos.keys())
                ni = idx_list[j]
                # use subplanner to test
                subprm = self.SubPRMClass(self._collisionChecker)
                subpath = subprm.planPath([p], [nodes_pos[ni]], prm_args)
                if subpath:
                    w = math.dist(p, nodes_pos[ni])
                    self.graph.add_edge(label, ni, weight=w)
                    break  # connect once
        # find shortest weighted path
        try:
            path = nx.shortest_path(self.graph, 'start', 'goal', weight='weight')
        except nx.NetworkXNoPath:
            return []
        return path

    def get_path_coordinates(self, high_path):
        coords = []
        prm_args = self._visConfig
        # reconstruct detailed path segments
        nodes_pos = nx.get_node_attributes(self.graph, 'pos')
        for u, v in zip(high_path[:-1], high_path[1:]):
            p1 = nodes_pos[u];
            p2 = nodes_pos[v]
            subprm = self.SubPRMClass(self._collisionChecker)
            subpath = subprm.planPath([p1], [p2], prm_args)
            # get subplanner graph positions if possible
            try:
                segment = [subprm.graph.nodes[n]['pos'] for n in subpath]
            except Exception:
                segment = [p1, p2]
            if coords and segment:
                segment = segment[1:]
            coords.extend(segment)
        return coords
