# coding: utf-8
"""
Visibility PRM with Lazy PRM as internal planner (enhanced)
"""
from lectures.IPLazyPRM import LazyPRM
from HierarchicalPRMBase import HierarchicalPRMBase
from lectures.IPVisibilityPRM import VisibilityStatsHandler


class VisPRM_with_LazyPRM(HierarchicalPRMBase):
    def __init__(self, _collChecker):
        internal_planner = LazyPRM(_collChecker)
        super(VisPRM_with_LazyPRM, self).__init__(_collChecker, internal_planner)

    def _getInternalPlannerConfig(self):
        return {
            "initialRoadmapSize": 15,
            "updateRoadmapSize": 5,
            "kNearest": 5,
            "maxIterations": 10,
            "debug": True
        }