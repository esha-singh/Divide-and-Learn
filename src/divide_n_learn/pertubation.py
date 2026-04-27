import numpy as np
import torch
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Callable, Optional, Set

# ============================================================================
# PURE MATHEMATICAL FRAMEWORK - EDIT DISTANCE METRIC SPACE
# ============================================================================

class MinimalEditOracle(ABC):
    """
    Pure abstract definition of edit distance metric on solution space.
    NO problem-specific operations - only metric space structure.
    """
    
    @abstractmethod
    def compute_edit_distance(self, solution1: List, solution2: List) -> int:
        """
        Compute the minimal number of atomic edits to transform solution1 to solution2.
        This defines the metric d(s1, s2) on solution space.
        """
        pass
    
    @abstractmethod
    def get_neighbors_at_distance(self, solution: List, distance: int, 
                                 max_neighbors: Optional[int] = None) -> List[List]:
        """
        Get all solutions at exactly 'distance' edits from current solution.
        For efficiency, can limit to max_neighbors.
        """
        pass
    
    @abstractmethod
    def sample_neighbor_at_distance(self, solution: List, distance: int) -> Optional[List]:
        """
        Sample ONE uniformly random neighbor at given edit distance.
        Returns None if no such neighbor exists.
        """
        pass
    
    @abstractmethod
    def estimate_diameter(self) -> int:
        """
        Estimate the diameter of the solution space (max edit distance between any two solutions).
        Used for normalization.
        """
        pass


def create_problem_agnostic_improver(problem_type: str, problem_size: int) -> Callable:
    def improve_subproblem_oco_universal(self, solution: List[int], 
                                         subproblem_indices: List[int]) -> Tuple[List[int], float]:
        
        # Detect domain type
        if all(v in [0, 1] for v in solution):
            domain_type = "binary"
        elif set(solution) == set(range(len(solution))):
            domain_type = "permutation"
        else:
            domain_type = "categorical"
        
        # Feasibility checker
        def is_feasible(sol):
            if domain_type == "binary" and hasattr(self, 'item_weights'):
                weight = sum(self.item_weights[i] * sol[i] for i in range(len(sol)))
                return weight <= self.knapsack_capacity
            return True
        
        def generate_perturbation(sol, indices):
            """Generate domain-appropriate perturbation"""
            perturbed = sol.copy()
            
            if domain_type == "binary":
                if hasattr(self, 'item_weights'):
                    # Smart flip for knapsack
                    i = random.choice(indices) if indices else random.randint(0, len(sol)-1)
                    if perturbed[i] == 0:
                        new_weight = sum(self.item_weights[j] * perturbed[j] for j in range(len(perturbed)))
                        new_weight += self.item_weights[i]
                        if new_weight <= self.knapsack_capacity:
                            perturbed[i] = 1
                    else:
                        perturbed[i] = 0
                else:
                    # Simple bit flip
                    if indices:
                        i = random.choice(indices)
                        perturbed[i] = 1 - perturbed[i]
                        
            elif domain_type == "categorical":
                if indices and hasattr(self, 'bounds'):
                    pos = random.choice(indices)
                    valid = self.bounds[pos]
                    if isinstance(valid, list):
                        current = sol[pos]
                        alts = [v for v in valid if v != current]
                        if alts:
                            perturbed[pos] = random.choice(alts)
                            
            else:  # permutation
                if len(indices) >= 2:
                    i, j = random.sample(indices, 2)
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            
            return perturbed
        
        def move_toward(current, target, indices, num_ops):
            """Move current solution toward target"""
            result = current.copy()
            
            if domain_type == "binary":
                diffs = [i for i in indices if i < len(current) and current[i] != target[i]]
                for i in diffs[:num_ops]:
                    test = result.copy()
                    test[i] = target[i]
                    if is_feasible(test):
                        result[i] = target[i]
                        
            elif domain_type == "categorical":
                # Direct value copying for categorical
                diffs = [i for i in indices if i < len(current) and current[i] != target[i]]
                random.shuffle(diffs)
                for i in diffs[:num_ops]:
                    result[i] = target[i]
                    
            else:  # permutation
                diffs = [(i, current[i], target[i]) for i in indices 
                        if i < len(current) and i < len(target) and current[i] != target[i]]
                for _ in range(min(num_ops, len(diffs))):
                    if diffs:
                        idx, curr_val, target_val = random.choice(diffs)
                        if target_val in result:
                            target_pos = result.index(target_val)
                            result[idx], result[target_pos] = result[target_pos], result[idx]
                        diffs = [(i, c, t) for i, c, t in diffs if i != idx]
            
            return result
        
        def move_random(sol, indices, num_ops):
            """Random exploration move"""
            result = sol.copy()
            
            if domain_type == "binary":
                for _ in range(min(num_ops, len(indices))):
                    if indices:
                        test = result.copy()
                        i = random.choice(indices)
                        test[i] = 1 - test[i]
                        if is_feasible(test):
                            result = test
                            
            elif domain_type == "categorical":
                for _ in range(min(num_ops, len(indices))):
                    if indices and hasattr(self, 'bounds'):
                        pos = random.choice(indices)
                        valid = self.bounds[pos]
                        if isinstance(valid, list):
                            current = result[pos]
                            alts = [v for v in valid if v != current]
                            if alts:
                                result[pos] = random.choice(alts)
                                
            else:  # permutation
                for _ in range(min(num_ops, len(indices) // 2)):
                    if len(indices) >= 2:
                        i, j = random.sample(indices, 2)
                        if i < len(result) and j < len(result):
                            result[i], result[j] = result[j], result[i]
            
            return result
        
        # Main OCO loop
        current_solution = solution.copy()
        current_value = self.evaluate_fn(current_solution)
        best_solution = current_solution
        best_value = current_value
        
        T = min(self.nb_rounds if hasattr(self, 'nb_rounds') else 10, len(subproblem_indices))
        eta_0 = 1.0 / np.sqrt(T + 1)
        delta = 1.0 / max(1, problem_size)
        
        momentum = 0.9
        velocity = 0.0
        
        for t in range(T):
            eta_t = eta_0 / np.sqrt(t + 1)
            
            # Generate perturbation
            perturbation = generate_perturbation(current_solution, subproblem_indices)
            
            if perturbation == current_solution:
                continue
            
            # Gradient estimate
            perturbed_value = self.evaluate_fn(perturbation)
            gradient_estimate = (perturbed_value - current_value) / delta
            
            # Update velocity with momentum
            velocity = momentum * velocity + (1 - momentum) * gradient_estimate
            
            # Adaptive step size based on gradient magnitude
            gradient_magnitude = abs(velocity)
            if gradient_magnitude > 0.1:
                num_operations = 1
            elif gradient_magnitude < 0.01:
                num_operations = min(3, len(subproblem_indices) // 2)
            else:
                num_operations = 2
            
            # Apply move based on gradient direction
            if gradient_estimate > 0:
                new_solution = move_toward(current_solution, perturbation, 
                                          subproblem_indices, num_operations)
            else:
                new_solution = move_random(current_solution, 
                                          subproblem_indices, num_operations)
            
            # Evaluate and update if feasible
            if new_solution != current_solution and is_feasible(new_solution):
                new_value = self.evaluate_fn(new_solution)
                
                current_solution = new_solution
                current_value = new_value
                
                if new_value > best_value:
                    best_solution = new_solution
                    best_value = new_value
        
        return best_solution, best_value - self.evaluate_fn(solution)
    
    return improve_subproblem_oco_universal