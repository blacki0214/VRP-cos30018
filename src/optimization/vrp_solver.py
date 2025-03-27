import time
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

from src.models.route import Route
from src.optimization.base_optimizer import BaseOptimizer
from src.optimization.ga_optimizer import GAOptimizer
from src.optimization.or_ml_optimizer import ORToolsMLOptimizer
from src.optimization.ga_or_optimizer import GAOROptimizer
from src.optimization.ga_or_mod_optimizer import GAORModifiedOptimizer

class VRPSolver:
    """
    Main solver that coordinates all four optimization methods:
    1. Pure GA (Genetic Algorithm)
    2. OR-Tools with ML
    3. GA + OR without fitness modification
    4. GA + OR with modified fitness
    
    It compares and selects the best solution based on a consistent fitness evaluation.
    """
    
    def __init__(self, data_processor):
        """
        Initialize the VRP Solver with a data processor.
        
        Args:
            data_processor: DataProcessor containing problem data
        """
        self.data_processor = data_processor
        
        # Initialize the four optimizers
        self.ga_optimizer = GAOptimizer(data_processor)
        self.or_ml_optimizer = ORToolsMLOptimizer(data_processor)
        self.ga_or_optimizer = GAOROptimizer(data_processor)
        self.ga_or_mod_optimizer = GAORModifiedOptimizer(data_processor)
        
        # Solver results
        self.results = {
            'ga': {'solution': None, 'fitness': 0, 'metrics': None},
            'or_ml': {'solution': None, 'fitness': 0, 'metrics': None},
            'ga_or': {'solution': None, 'fitness': 0, 'metrics': None},
            'ga_or_mod': {'solution': None, 'fitness': 0, 'metrics': None}
        }
        
        # Best solution
        self.best_solution = None
        self.best_fitness = 0.0
        self.best_method = None
    
    def optimize(self, method: str = None) -> List[Route]:
        """
        Optimize VRP problem using specified method.
        
        Args:
            method: Optimization method to use. If None, will run all methods.
            
        Returns:
            Optimized solution (routes)
        """
        self.results = {}
        
        # Run specified method(s)
        if method is None or method == 'all':
            # Run all methods
            print("Running all optimization methods for comparison...")
            self._run_ga()
            self._run_or_ml()
            self._run_ga_or()
            self._run_ga_or_mod()
            
            # Return solution from best method
            best_method = max(self.results, 
                            key=lambda k: self.results[k]['scores']['composite_score'])
            return self.results[best_method]['solution']
            
        elif method == 'ga':
            # Genetic Algorithm
            print("Running GA optimization...")
            return self._run_ga()
            
        elif method == 'or_ml':
            # OR-Tools with ML
            print("Running OR-Tools with ML optimization...")
            return self._run_or_ml()
            
        elif method == 'ga_or':
            # GA + OR-Tools (No Fitness Mod)
            print("Running GA + OR-Tools (No Fitness Mod) optimization...")
            return self._run_ga_or()
            
        elif method == 'ga_or_mod':
            # GA + OR-Tools (Modified Fitness)
            print("Running GA + OR-Tools (Modified Fitness) optimization...")
            return self._run_ga_or_mod()
            
        else:
            # Unknown method
            raise ValueError(f"Unknown method: {method}")
    
    def _update_results(self, method: str, solution: List[Route], optimizer: BaseOptimizer):
        """
        Update results for a single method.
        
        Args:
            method: Method name
            solution: Optimized solution
            optimizer: Optimizer used
        """
        try:
            metrics = optimizer.evaluate_solution(solution)
            fitness = optimizer.calculate_fitness(solution)
            scores = optimizer.calculate_route_comparison_score(solution)
            self.results[method] = {
                'solution': solution,
                'metrics': metrics,
                'fitness': fitness,
                'scores': scores
            }
            
            # Update the visualization with method comparison scores if we have multiple methods
            if len(self.results) > 1 and hasattr(self, 'main_window') and self.main_window and hasattr(self.main_window, 'visualizer'):
                method_scores = {m: self.results[m]['scores']['composite_score'] for m in self.results}
                self.main_window.visualizer.update_method_comparison(method_scores)
        except Exception as e:
            print(f"Error updating results: {str(e)}")
    
    def _get_optimizer(self, method: str) -> BaseOptimizer:
        """
        Get optimizer for a specified method.
        
        Args:
            method: Method name
            
        Returns:
            Optimizer for the method
        """
        if method == 'ga':
            return self.ga_optimizer
        elif method == 'or_ml':
            return self.or_ml_optimizer
        elif method == 'ga_or':
            return self.ga_or_optimizer
        elif method == 'ga_or_mod':
            return self.ga_or_mod_optimizer
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _print_comparison(self):
        """Print comparison of all methods"""
        print("\n=== Results Comparison ===")
        
        # Calculate comparison scores for each method
        method_scores = {}
        for method_name, result in self.results.items():
            metrics = result['metrics']
            fitness = result['fitness']
            comparison_scores = self.ga_optimizer.calculate_route_comparison_score(result['solution'])
            
            print(f"\n{self._get_method_display_name(method_name)}:")
            print(f"  Composite Score: {comparison_scores['composite_score']:.4f}")
            print(f"  Detailed Scores:")
            print(f"    - Distance Efficiency: {comparison_scores['distance_score']:.4f}")
            print(f"    - Cost Efficiency: {comparison_scores['cost_efficiency_score']:.4f}")
            print(f"    - Capacity Utilization: {comparison_scores['capacity_utilization_score']:.4f}")
            print(f"    - Parcel Efficiency: {comparison_scores['parcel_efficiency_score']:.4f}")
            print(f"    - Route Structure: {comparison_scores['route_structure_score']:.4f}")
            print(f"\n  Route Metrics:")
            print(f"    - Parcels: {metrics['parcels_delivered']}")
            print(f"    - Cost: ${metrics['total_cost']:.2f}")
            print(f"    - Distance: {metrics['total_distance']:.2f} km")
            print(f"    - Routes: {metrics['num_routes']}")
            print(f"    - Avg Load Factor: {metrics['avg_load_factor']*100:.1f}%")
            
            method_scores[method_name] = comparison_scores['composite_score']
        
        # Determine best method based on composite score
        best_method = max(method_scores.items(), key=lambda x: x[1])[0]
        print(f"\nBest method: {self._get_method_display_name(best_method)} with composite score {method_scores[best_method]:.4f}")
        
        # Update the visualization with method comparison scores
        if hasattr(self, 'main_window') and self.main_window and hasattr(self.main_window, 'visualizer'):
            self.main_window.visualizer.update_method_comparison(method_scores)
    
    def _print_metrics(self, solution: List[Route]):
        """
        Print metrics for a solution.
        
        Args:
            solution: Solution to evaluate
        """
        metrics = self.ga_optimizer.evaluate_solution(solution)
        
        print(f"Parcels Delivered: {metrics['parcels_delivered']}")
        print(f"Total Cost: ${metrics['total_cost']:.2f}")
        print(f"Total Distance: {metrics['total_distance']:.2f} km")
        print(f"Number of Routes: {metrics['num_routes']}")
        print(f"Average Load Factor: {metrics['avg_load_factor']*100:.1f}%")
    
    def _save_comparison_chart(self):
        """Create and save comparison chart of all methods"""
        method_names = [self._get_method_display_name(m) for m in self.results.keys()]
        fitness_scores = [result['fitness'] for result in self.results.values()]
        parcels = [result['metrics']['parcels_delivered'] for result in self.results.values()]
        costs = [result['metrics']['total_cost'] for result in self.results.values()]
        distances = [result['metrics']['total_distance'] for result in self.results.values()]
        routes = [result['metrics']['num_routes'] for result in self.results.values()]
        
        # Create figure and axes
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('VRP Optimization Methods Comparison', fontsize=16)
        
        # Fitness scores
        axs[0, 0].bar(method_names, fitness_scores, color='skyblue')
        axs[0, 0].set_title('Fitness Scores')
        axs[0, 0].set_ylabel('Score (higher is better)')
        plt.setp(axs[0, 0].get_xticklabels(), rotation=45, ha='right')
        
        # Parcels delivered
        axs[0, 1].bar(method_names, parcels, color='lightgreen')
        axs[0, 1].set_title('Parcels Delivered')
        axs[0, 1].set_ylabel('Count')
        plt.setp(axs[0, 1].get_xticklabels(), rotation=45, ha='right')
        
        # Total cost
        axs[1, 0].bar(method_names, costs, color='salmon')
        axs[1, 0].set_title('Total Cost')
        axs[1, 0].set_ylabel('Cost ($)')
        plt.setp(axs[1, 0].get_xticklabels(), rotation=45, ha='right')
        
        # Number of routes
        axs[1, 1].bar(method_names, routes, color='plum')
        axs[1, 1].set_title('Number of Routes')
        axs[1, 1].set_ylabel('Count')
        plt.setp(axs[1, 1].get_xticklabels(), rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Save figure
        plt.savefig('results/methods_comparison.png')
        print("Comparison chart saved to 'results/methods_comparison.png'")
    
    def _get_method_display_name(self, method: str) -> str:
        """
        Get display name for a method.
        
        Args:
            method: Method key
            
        Returns:
            Display name
        """
        display_names = {
            'ga': 'Pure GA',
            'or_ml': 'OR-Tools + ML',
            'ga_or': 'GA + OR (no mod)',
            'ga_or_mod': 'GA + OR (mod fitness)'
        }
        return display_names.get(method, method)
    
    def stop(self):
        """Stop all optimization processes"""
        self.ga_optimizer.stop()
        self.ga_or_optimizer.stop()
        self.ga_or_mod_optimizer.stop()

    def set_main_window(self, main_window):
        """Set the main window reference for score updates"""
        self.main_window = main_window

    def _run_ga(self) -> List[Route]:
        """Run Pure GA optimization"""
        print("\n--- Method: Pure GA ---")
        solution = self.ga_optimizer.optimize()
        self._update_results('ga', solution, self.ga_optimizer)
        
        # Update main window scores if available
        if hasattr(self, 'main_window'):
            scores = self.ga_optimizer.calculate_route_comparison_score(solution)
            self.main_window.update_scores(scores)
            
        return solution
    
    def _run_or_ml(self) -> List[Route]:
        """Run OR-Tools with ML optimization"""
        print("\n--- Method: OR-Tools with ML ---")
        solution = self.or_ml_optimizer.optimize()
        self._update_results('or_ml', solution, self.ga_optimizer)
        
        # Update main window scores if available
        if hasattr(self, 'main_window'):
            scores = self.ga_optimizer.calculate_route_comparison_score(solution)
            self.main_window.update_scores(scores)
            
        return solution
    
    def _run_ga_or(self) -> List[Route]:
        """Run GA + OR (no fitness modification) optimization"""
        print("\n--- Method: GA + OR (no fitness modification) ---")
        solution = self.ga_or_optimizer.optimize()
        self._update_results('ga_or', solution, self.ga_optimizer)
        
        # Update main window scores if available
        if hasattr(self, 'main_window'):
            scores = self.ga_optimizer.calculate_route_comparison_score(solution)
            self.main_window.update_scores(scores)
            
        return solution
    
    def _run_ga_or_mod(self) -> List[Route]:
        """Run GA + OR with modified fitness optimization"""
        print("\n--- Method: GA + OR with modified fitness ---")
        solution = self.ga_or_mod_optimizer.optimize()
        self._update_results('ga_or_mod', solution, self.ga_optimizer)
        
        # Update main window scores if available
        if hasattr(self, 'main_window'):
            scores = self.ga_optimizer.calculate_route_comparison_score(solution)
            self.main_window.update_scores(scores)
            
        return solution

    def _update_visualization(self, solution: List[Route]):
        """Update visualization with final solution"""
        if self.main_window.visualizer:
            # Plot final routes
            self.main_window.visualizer.plot_routes(solution)
            
            # Update method comparison if we have method scores
            if hasattr(self, 'results') and self.results:
                method_scores = {m: result['scores']['composite_score'] 
                                for m, result in self.results.items() 
                                if 'scores' in result and 'composite_score' in result['scores']}
                
                if method_scores:
                    self.main_window.visualizer.update_method_comparison(method_scores)