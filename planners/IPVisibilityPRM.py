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
