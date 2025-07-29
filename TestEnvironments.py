# coding: utf-8
"""
Test environments for hierarchical planning
"""
from shapely.geometry import Point, Polygon, LineString
from shapely import plotting
import numpy as np

def create_point_robot_environments():
    """Create 2D point robot test environments"""
    environments = {}
    
    # Maze environment
    maze = {}
    maze["wall1"] = LineString([(1,0), (1,8)]).buffer(0.2)
    maze["wall2"] = LineString([(3,2), (3,10)]).buffer(0.2)
    maze["wall3"] = LineString([(5,0), (5,6)]).buffer(0.2)
    maze["wall4"] = LineString([(7,4), (7,10)]).buffer(0.2)
    environments["maze"] = maze
    
    # Cluttered environment
    cluttered = {}
    for i in range(15):
        x, y = np.random.uniform(0, 10, 2)
        cluttered[f"obs{i}"] = Point(x,y).buffer(0.3)
    environments["cluttered"] = cluttered
    
    # Narrow passage
    narrow = {}
    narrow["left"] = LineString([(0,0), (0,10)]).buffer(0.8)
    narrow["right"] = LineString([(2,0), (2,10)]).buffer(0.8)
    narrow["block"] = Polygon([(0.8,4), (1.2,4), (1.2,6), (0.8,6)])
    environments["narrow_passage"] = narrow
    
    return environments

def create_planar_robot_environments():
    """Create 2-DoF planar robot environments"""
    environments = {}
    
    # Central obstacle
    central = {}
    central["obs"] = Polygon([(3,3), (7,3), (7,7), (3,7)])
    environments["central_obstacle"] = central
    
    # Multiple narrow passages
    passages = {}
    passages["wall1"] = LineString([(2,0), (2,5)]).buffer(0.2)
    passages["wall2"] = LineString([(4,5), (4,10)]).buffer(0.2)
    passages["wall3"] = LineString([(6,0), (6,5)]).buffer(0.2)
    passages["wall4"] = LineString([(8,5), (8,10)]).buffer(0.2)
    environments["multiple_passages"] = passages
    
    return environments

def create_high_dof_environments(dof):
    """Create environments for high-DoF robots"""
    environments = {}
    
    # Complex layered environment
    layered = {}
    for i in range(dof):
        y = 2 + i * 6/dof
        layered[f"layer{i}"] = LineString([(1,y), (9,y)]).buffer(0.1)
    environments["layered"] = layered
    
    # Random obstacles
    random_obs = {}
    for i in range(dof*2):
        x, y = np.random.uniform(1, 9, 2)
        random_obs[f"obs{i}"] = Point(x,y).buffer(0.3)
    environments["random_obstacles"] = random_obs
    
    return environments