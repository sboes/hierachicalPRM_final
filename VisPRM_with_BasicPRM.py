# coding: utf-8
"""
Visibility PRM with Basic PRM as internal planner (enhanced)
"""
from lectures.IPBasicPRM import BasicPRM
from HierarchicalPRMBase import HierarchicalPRMBase
from lectures.IPVisibilityPRM import VisibilityStatsHandler


class VisPRM_with_BasicPRM(HierarchicalPRMBase):
    def __init__(self, _collChecker):
        internal_planner = BasicPRM(_collChecker)
        super(VisPRM_with_BasicPRM, self).__init__(_collChecker, internal_planner)

    def _getInternalPlannerConfig(self):
        return {
            "radius": 1.0,
            "numNodes": 20,
            "useKDTree": True,
            "debug": True
        }