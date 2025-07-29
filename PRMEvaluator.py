# coding: utf-8
"""
PRM Evaluation Framework
"""
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from collections import defaultdict
from lectures.IPEnvironment import CollisionChecker

class PRMEvaluator:
    def __init__(self, planners, environments):
        self.planners = planners
        self.environments = environments
        self.results = defaultdict(list)
        
    def evaluate(self, num_trials=5):
        """Run evaluation across all planners and environments"""
        for env_name, env in self.environments.items():
            for planner_name, planner in self.planners.items():
                for trial in range(num_trials):
                    self._run_single_test(env_name, planner_name, planner, env)
                    
        self._analyze_results()
        
    def _run_single_test(self, env_name, planner_name, planner, env):
        """Run a single test case"""
        coll_checker = CollisionChecker(env)
        planner_instance = planner(coll_checker)
        
        # Generate random start and goal
        start = self._get_random_free_position(coll_checker)
        goal = self._get_random_free_position(coll_checker)
        
        # Run planning
        start_time = time.time()
        path = planner_instance.planPath([start], [goal], self._get_planner_config(planner_name))
        planning_time = time.time() - start_time
        
        # Store results
        result = {
            'success': len(path) > 0,
            'time': planning_time,
            'path_length': self._calculate_path_length(path, planner_instance) if path else 0,
            'roadmap_size': len(planner_instance.graph.nodes()),
            'path_nodes': len(path) if path else 0
        }
        
        self.results[(env_name, planner_name)].append(result)
        
    def _get_planner_config(self, planner_name):
        """Get appropriate config for each planner type"""
        if "Basic" in planner_name:
            return {"radius": 2.0, "numNodes": 100}
        elif "Lazy" in planner_name:
            return {"initialRoadmapSize": 50, "updateRoadmapSize": 20, 
                    "kNearest": 10, "maxIterations": 30}
        else:  # Visibility variants
            return {"ntry": 50}
            
    def _analyze_results(self):
        """Analyze and visualize results"""
        # Aggregate results
        aggregated = {}
        for key, trials in self.results.items():
            env, planner = key
            if planner not in aggregated:
                aggregated[planner] = {}
                
            success_rate = sum(t['success'] for t in trials) / len(trials)
            avg_time = np.mean([t['time'] for t in trials if t['success']])
            avg_path_len = np.mean([t['path_length'] for t in trials if t['success']])
            avg_roadmap = np.mean([t['roadmap_size'] for t in trials])
            
            aggregated[planner][env] = {
                'success_rate': success_rate,
                'avg_time': avg_time,
                'avg_path_len': avg_path_len,
                'avg_roadmap': avg_roadmap
            }
        
        # Visualization
        self._plot_metrics(aggregated)
        
    def _plot_metrics(self, data):
        """Plot comparison metrics"""
        metrics = ['success_rate', 'avg_time', 'avg_path_len', 'avg_roadmap']
        titles = ['Success Rate', 'Planning Time (s)', 'Path Length', 'Roadmap Size']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            planners = list(data.keys())
            environments = list(next(iter(data.values())).keys())
            
            x = np.arange(len(environments))
            width = 0.8 / len(planners)
            
            for j, planner in enumerate(planners):
                values = [data[planner][env][metric] for env in environments]
                ax.bar(x + j*width, values, width, label=planner)
                
            ax.set_title(titles[i])
            ax.set_xticks(x + width*(len(planners)-1)/2)
            ax.set_xticklabels(environments, rotation=45)
            ax.legend()
            
        plt.tight_layout()
        plt.show()
        
    def _get_random_free_position(self, coll_checker):
        """Generate random free position"""
        limits = coll_checker.getEnvironmentLimits()
        while True:
            pos = [np.random.uniform(l[0], l[1]) for l in limits]
            if not coll_checker.pointInCollision(pos):
                return pos
                
    def _calculate_path_length(self, path, planner):
        """Calculate path length"""
        if not path or len(path) < 2:
            return 0
            
        length = 0
        for i in range(len(path)-1):
            pos1 = planner.graph.nodes[path[i]]['pos']
            pos2 = planner.graph.nodes[path[i+1]]['pos']
            length += np.linalg.norm(np.array(pos1) - np.array(pos2))
            
        return length