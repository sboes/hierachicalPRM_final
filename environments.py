
import random
from shapely.geometry import Polygon
from hierarchical_planner_full import Workspace2D

def env_maze():
    obs = [
        Polygon([(0.2,0.2),(0.8,0.2),(0.8,0.25),(0.2,0.25)]),
        Polygon([(0.2,0.5),(0.6,0.5),(0.6,0.55),(0.2,0.55)]),
        Polygon([(0.42,0.25),(0.48,0.25),(0.48,0.5),(0.42,0.5)]),
        Polygon([(0.45,0.55),(0.5,0.55),(0.5,0.8),(0.45,0.8)]),
    ]
    return Workspace2D(obs,(0,0,1,1))

def env_bugtrap():
    u = Polygon([(0.2,0.2),(0.8,0.2),(0.8,0.25),(0.25,0.25),(0.25,0.75),(0.8,0.75),(0.8,0.8),(0.2,0.8)])
    gap = Polygon([(0.45,0.2),(0.55,0.2),(0.55,0.25),(0.45,0.25)])
    u = u.difference(gap)
    return Workspace2D([u], (0,0,1,1))

def env_clutter():
    rng = random.Random(42); obs=[]
    for _ in range(8):
        x=rng.uniform(0.05,0.85); y=rng.uniform(0.05,0.85)
        w=rng.uniform(0.06,0.14); h=rng.uniform(0.06,0.14)
        obs.append(Polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)]))
    return Workspace2D(obs,(0,0,1,1))

def make_manip_checker(dof: int):
    # Optional: you can plug in your KinChain here
    from hierarchical_planner_full import CollisionCheckerBase
    class Dummy(CollisionCheckerBase):
        def dimension(self): return dof
        def is_free(self,q): return True
    return Dummy()
