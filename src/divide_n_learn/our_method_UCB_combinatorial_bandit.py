
### file improved over original (from internet)
import numpy as np
import torch
import random
from typing import List, Tuple, Dict, Optional, Callable
from collections import defaultdict


from .pertubation import create_problem_agnostic_improver
from collections import deque

CUDA_DEVICE = -1
class FixedDecomposedBanditUCBHedge:
    def __init__(self, 
                 problem_size: int,
                 evaluate_fn: Callable,
                 learning_rate: float = 0.5,
                 ucb_coefficient: float = 1.0,
                 initial_temperature: float = 1.0,
                 temp_decay: float = 0.98,
                 hybrid_ratio: float = 0.5,
                 adaptive_hybrid: bool = True,
                 decomposition_size: int = 10,
                 overlap: int = 3,
                 max_iterations: int = 0,
                 nb_rounds: int = 5,
                 patience: int = 500, # 50
                 reference_point: Optional[Tuple[float, float]] = None,
                 use_lagrangian: bool = True,
                 use_ftrl: bool = True,
                 ftrl_rate: float = 0.3,  # NEW
                 dual_step_size: float = 0.1,
                 use_accelerated_dual: bool = True,  # NEW: Enable accelerated dual updates
                 use_diminishing_overlap: bool = True,  # NEW: Enable diminishing overlap
                 overlap_decay_rate: float = 0.5,  # NEW: Decay rate for overlap (alpha parameter)
                 item_weights=None,  # Add this
                 knapsack_capacity=None,  # Add this
                 problem_type=None,  # Add this
                 **kwargs):
        

        max_iterations = 400 # 300 # 80
        
        # Store the metadata
        self.problem_size = problem_size
        self.item_weights = item_weights # np.random.uniform(0, 1, size=self.problem_size)
        self.knapsack_capacity = kwargs.get('knapsack_capacity', 25) 
        self.problem_type = kwargs.get('problem_type', 'Knapsack')

        # self.problem_size = problem_size
        self.evaluate_fn = evaluate_fn
        self.learning_rate = learning_rate
        self.ucb_coefficient = ucb_coefficient
        self.temperature = initial_temperature
        self.temp_decay = temp_decay
        self.hybrid_ratio = hybrid_ratio
        self.adaptive_hybrid = adaptive_hybrid
        self.decomposition_size = min(decomposition_size, problem_size)
        self.initial_overlap = min(overlap, decomposition_size - 1)  # Store initial overlap
        self.overlap = self.initial_overlap  # Current overlap (will change if diminishing)
        self.max_iterations = max_iterations
        self.nb_rounds = nb_rounds
        self.patience = patience
        self.reference_point = reference_point
        self.use_lagrangian = use_lagrangian
        self.use_ftrl = use_ftrl
        self.ftrl_rate = ftrl_rate  # NEW
        self.dual_step_size = dual_step_size
        self.use_accelerated_dual = use_accelerated_dual  # NEW
        self.use_diminishing_overlap = use_diminishing_overlap  # NEW
        self.overlap_decay_rate = overlap_decay_rate  # NEW
        
        # GLOBAL parameters shared across all subproblems
        self.value_estimates = torch.zeros((problem_size, problem_size))
        self.visit_counts = torch.zeros((problem_size, problem_size))
        self.weights = torch.ones((problem_size, problem_size))
        
        # FTRL parameters
        if self.use_ftrl:
            self.sum_losses = torch.zeros((problem_size, problem_size))
            self.ftrl_regularizer = 1.0 / np.sqrt(problem_size)
        
        # Lagrangian dual variables for overlapping positions
        if self.use_lagrangian:
            self.dual_vars = torch.zeros(problem_size)
            self.dual_momentum = torch.zeros(problem_size)
            
            # NEW: For accelerated dual updates (Nesterov momentum)
            if self.use_accelerated_dual:
                self.dual_vars_prev = torch.zeros(problem_size)  # Previous dual variables
                self.dual_velocity = torch.zeros(problem_size)  # Velocity term for acceleration
                self.dual_iteration = 0  # Track dual update iterations
        
        # Create initial decomposition
        self.subproblems = self._create_simple_decomposition()
        self.n_subproblems = len(self.subproblems)
        
        # Track which subproblems contain each position
        self.position_to_subproblems = defaultdict(list)
        for sp_idx, subproblem in enumerate(self.subproblems):
            for pos in subproblem:
                self.position_to_subproblems[pos].append(sp_idx)
        
        # Global tracking
        self.total_iterations = 0
        self.solutions = []
        self.rewards = []
        self.normalized_rewards = []
        self.best_solution = None
        self.best_reward = float('-inf')
        self.min_reward = float('inf')
        self.max_reward = float('-inf')
        
        # For adaptive behavior
        self.uncertainty_history = []
        self.no_improvement_count = 0
    
    def _create_simple_decomposition(self) -> List[List[int]]:
        """Create simple sliding window decomposition."""
        if self.decomposition_size >= self.problem_size:
            return [list(range(self.problem_size))]
        
        subproblems = []
        step = max(1, self.decomposition_size - self.overlap)
        
        for i in range(0, self.problem_size, step):
            end = min(i + self.decomposition_size, self.problem_size)
            window = list(range(i, end))
            
            if end == self.problem_size and len(window) < self.decomposition_size:
                remaining = self.decomposition_size - len(window)
                window.extend(list(range(min(remaining, self.overlap))))
            
            subproblems.append(window)
        
        if self.problem_size > self.decomposition_size:
            wraparound = list(range(self.problem_size - self.overlap, self.problem_size))
            wraparound.extend(list(range(self.overlap)))
            if len(wraparound) >= self.decomposition_size // 2:
                subproblems.append(wraparound)
        
        return subproblems
    
    def _update_overlap(self):
        """NEW: Update overlap based on diminishing schedule"""
        if not self.use_diminishing_overlap:
            return
        
        # Diminishing overlap: overlap_t = overlap_0 * (1 / t^alpha)
        if self.total_iterations > 1:
            decay_factor = 1.0 / (self.total_iterations ** self.overlap_decay_rate)
            new_overlap = max(0, int(self.initial_overlap * decay_factor))
            
            # Only update if overlap actually changed
            if new_overlap != self.overlap:
                self.overlap = new_overlap
                # Recreate decomposition with new overlap
                self.subproblems = self._create_simple_decomposition()
                self.n_subproblems = len(self.subproblems)
                
                # Update position mappings
                self.position_to_subproblems = defaultdict(list)
                for sp_idx, subproblem in enumerate(self.subproblems):
                    for pos in subproblem:
                        self.position_to_subproblems[pos].append(sp_idx)
    
    def update_dual_variables(self, solution: List[int]):
        """Update Lagrangian dual variables with optional acceleration"""
        if not self.use_lagrangian:
            return
        
        # Track actual coordination conflicts between overlapping subproblems
        violations = torch.zeros(self.problem_size)
        
        for pos in range(self.problem_size):
            subproblems_containing_pos = self.position_to_subproblems[pos]
            if len(subproblems_containing_pos) > 1:
                assigned_value = solution[pos]
                
                value_variance = torch.var(self.value_estimates[pos])
                visit_ratio = self.visit_counts[pos, assigned_value] / (self.visit_counts[pos].sum() + 1e-10)
                
                violations[pos] = (len(subproblems_containing_pos) - 1) * value_variance.item() * (1 - visit_ratio.item())
        
        for pos in range(self.problem_size):
            if len(self.position_to_subproblems[pos]) > 1:
                violations[pos] += 0.1
        
        if self.use_accelerated_dual:
            # NEW: Nesterov accelerated mirror descent
            self.dual_iteration += 1
            
            theta = 2.0 / (self.dual_iteration + 1)
            step_size = self.dual_step_size / np.sqrt(self.dual_iteration)
            
            dual_vars_old = self.dual_vars.clone()
            
            # Mirror descent step with entropy regularization
            self.dual_vars = torch.maximum(self.dual_vars, torch.tensor(1e-10))
            log_dual = torch.log(self.dual_vars + 1e-10)
            log_dual_update = log_dual + step_size * violations
            
            self.dual_vars = torch.exp(log_dual_update)
            self.dual_vars = torch.clamp(self.dual_vars, min=0, max=10)
            
            # Nesterov acceleration
            self.dual_velocity = self.dual_vars + (1 - theta) * (self.dual_vars - self.dual_vars_prev)
            
            self.dual_vars_prev = dual_vars_old
            self.dual_vars = torch.clamp(self.dual_velocity, min=0, max=10)
        else:
            # Original subgradient update with momentum
            self.dual_momentum = 0.9 * self.dual_momentum + violations
            step_size = self.dual_step_size / np.sqrt(self.total_iterations + 1)
            self.dual_vars = torch.clamp(self.dual_vars + step_size * self.dual_momentum, min=0, max=10)
    
    def compute_uncertainty_for_positions(self, positions: List[int]) -> float:
        """Compute uncertainty for a set of positions using global parameters."""
        if not positions:
            return 1.0

        total_visits = 0
        for pos in positions:
            total_visits += self.visit_counts[pos].sum().item()

        if total_visits == 0:
            return 1.0

        avg_visits = total_visits / (len(positions) * self.problem_size)
        uncertainty = 1.0 / (1.0 + np.log(1.0 + avg_visits))

        return uncertainty

    def get_max_ucb_confidence_width(self) -> float:
        """
        Compute maximum UCB confidence interval width - theoretically grounded convergence metric.

        Based on UCB regret bound O(√(T log T)). When this value is small,
        we have tight confidence bounds around true action values.
        This is the ONLY convergence metric with PAC guarantees.

        Time: O(n) where n=problem_size (just check max visits per position)
        """
        if self.total_iterations == 0:
            return float('inf')

        max_conf = 0.0
        T = self.total_iterations + 2
        log_T = np.log(T)

        # Check most-visited action at each position
        for pos in range(self.problem_size):
            n = self.visit_counts[pos].max().item()
            if n > 0:
                conf = self.ucb_coefficient * np.sqrt(log_T / n)
                if conf > max_conf:
                    max_conf = conf

        return max_conf

    def get_adaptive_hybrid_ratio(self, positions: Optional[List[int]] = None) -> float:
        """Get adaptive hybrid ratio based on uncertainty."""
        if not self.adaptive_hybrid:
            return self.hybrid_ratio
        
        if positions is not None:
            uncertainty = self.compute_uncertainty_for_positions(positions)
        else:
            uncertainty = self.compute_uncertainty_for_positions(list(range(self.problem_size)))
        
        return self.hybrid_ratio * (1.0 - uncertainty * 0.5)
    
    def compute_lagrangian_modified_scores(self, position: int, available_values: List[int]) -> torch.Tensor:
        """Compute scores modified by Lagrangian dual variables."""
        base_scores = torch.zeros(len(available_values))
        
        for idx, val in enumerate(available_values):
            base_scores[idx] = self.value_estimates[position, val]
            
            if self.use_lagrangian:
                num_subproblems = len(self.position_to_subproblems[position])
                if num_subproblems > 1:
                    base_scores[idx] -= self.dual_vars[position] * (num_subproblems - 1) 
        
        return base_scores
    
    def select_action_for_position(self, position: int, available_values: List[int]) -> int:
        """Select action using global UCB-Hedge parameters with FTRL and Lagrangian."""
        if not available_values:
            return random.choice(list(range(self.problem_size)))
        
        current_ratio = self.get_adaptive_hybrid_ratio([position])
        
        avail_mask = torch.zeros(self.problem_size, dtype=torch.bool)
        avail_mask[available_values] = True
        
        if self.use_ftrl and random.random() < self.ftrl_rate: #0.3
            ftrl_scores = torch.zeros(self.problem_size)
            
            for val in available_values:
                cumulative_loss = self.sum_losses[position, val]
                # regularization = self.ftrl_regularizer * torch.sqrt(
                #     self.visit_counts[position, val] + 1)
                
                # New
                # Entropic FTRL approximation: encourages exploration without UCB optimism
                regularization = -self.ftrl_regularizer * torch.log(self.visit_counts[position, val] + 1e-8)
                ftrl_scores[val] = -cumulative_loss + regularization
                
                if self.use_lagrangian:
                    num_subproblems = len(self.position_to_subproblems[position])
                    if num_subproblems > 1:
                        ftrl_scores[val] -= self.dual_vars[position] * (num_subproblems - 1) 
            
            ftrl_scores[~avail_mask] = float('-inf')
            value = available_values[torch.argmax(ftrl_scores[available_values]).item()]
            
        elif random.random() < current_ratio:
            pos_weights = self.weights[position].clone()
            
            if self.use_lagrangian:
                lagrangian_scores = self.compute_lagrangian_modified_scores(position, available_values)
                for idx, val in enumerate(available_values):
                    pos_weights[val] *= torch.exp(0.1 * lagrangian_scores[idx])
            
            temp = max(self.temperature, 1e-3)
            logits = torch.log(pos_weights + 1e-10) / temp
            logits[~avail_mask] = float('-inf')
            logits = torch.clamp(logits, min=-50, max=50)
            
            probs = torch.softmax(logits, dim=0)
            
            if torch.isnan(probs).any() or (probs < 0).any():
                return random.choice(available_values)
            
            try:
                value = torch.multinomial(probs, 1).item()
            except:
                value = random.choice(available_values)
        else:
            ucb_scores = torch.zeros(self.problem_size)
            
            lagrangian_scores = self.compute_lagrangian_modified_scores(position, available_values)
            
            for idx, val in enumerate(available_values):
                visit_count = max(1e-10, self.visit_counts[position, val].item())
                exploration = self.ucb_coefficient * torch.sqrt(
                    torch.log(torch.tensor(self.total_iterations + 1)) / visit_count)
                ucb_scores[val] = lagrangian_scores[idx] + exploration
            
            ucb_scores[~avail_mask] = float('-inf')
            value = torch.argmax(ucb_scores).item()
        
        return value
    

    # testing for normalization fix | gave good results
    def update_global_parameters(self, solution: List[int], reward: float):
        """Update global UCB and Hedge parameters with normalization.
        
        CORRECTED FOR BANDIT SETTING:
        - Uses importance weighting for EXP3 (not Hedge)
        - Uses importance weighting for Bandit FTRL
        - UCB updates unchanged (already correct for bandits)
        """
        
        # Initialize reward normalizer if not exists
        if not hasattr(self, 'reward_normalizer'):
            self.reward_normalizer = []
            self.reward_mean = 0.0
            self.reward_m2 = 0.0
            self.reward_count = 0
        
        # Update running statistics (Welford's algorithm)
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_m2 += delta * delta2
        
        # Store recent rewards for percentile normalization
        self.reward_normalizer.append(reward)
        if len(self.reward_normalizer) > 1000:
            self.reward_normalizer.pop(0)
        
        # Normalize reward
        if len(self.reward_normalizer) >= 10:
            # Use robust percentile normalization
            rewards_array = np.array(self.reward_normalizer)
            p10 = np.percentile(rewards_array, 10)
            p90 = np.percentile(rewards_array, 90)
            range_val = p90 - p10
            
            if range_val > 1e-10:
                normalized_reward = (reward - p10) / range_val
                normalized_reward = np.clip(normalized_reward, -2, 2)
            else:
                normalized_reward = 0.0
        else:
            # Fallback for early iterations
            if self.reward_count > 1:
                std_dev = np.sqrt(self.reward_m2 / self.reward_count)
                normalized_reward = (reward - self.reward_mean) / (std_dev + 1e-10) if std_dev > 0 else 0
                normalized_reward = np.tanh(normalized_reward)  # Squash to [-1, 1]
            else:
                normalized_reward = np.tanh(reward / 10.0)  # Initial guess
        
        # Update parameters
        self.total_iterations += 1
        
        # Compute loss for FTRL (in normalized scale)
        loss = 1.0 - normalized_reward
        
        for pos, val in enumerate(solution):
            # Update visit count
            self.visit_counts[pos, val] += 1
            n = self.visit_counts[pos, val]
            
            # IMPORTANCE WEIGHTING FOR BANDIT SETTING
            # Compute selection probability (needed for EXP3 and FTRL)
            weights_sum = self.weights[pos].sum()
            prob_selected = self.weights[pos, val] / (weights_sum + 1e-10)
            # Clip probability to prevent extreme importance weights
            prob_selected = max(prob_selected.item() if torch.is_tensor(prob_selected) else prob_selected, 0.01)
            
            # 1. UCB update (no importance weighting needed - running average is correct for bandits)
            old_estimate = self.value_estimates[pos, val]
            self.value_estimates[pos, val] = old_estimate + (normalized_reward - old_estimate) / n
            
            # 2. EXP3 update (CORRECTED from Hedge with importance weighting)
            # Importance-weighted reward estimate
            estimated_reward = normalized_reward / prob_selected
            
            # Update weight with importance-weighted estimate (scaled by problem size for stability)
            reward_tensor = torch.tensor(estimated_reward / self.problem_size, 
                                         device=self.weights.device, dtype=self.weights.dtype)
            self.weights[pos, val] *= torch.exp(self.learning_rate * reward_tensor)
            
            # 3. Bandit FTRL update (CORRECTED with importance weighting)
            if self.use_ftrl:
                # Importance-weighted loss estimate
                estimated_loss = loss / prob_selected
                # Clip to prevent explosion
                estimated_loss = min(estimated_loss, 100.0)
                self.sum_losses[pos, val] += estimated_loss
        
        # Normalize weights to prevent overflow
        for pos in range(self.problem_size):
            row_sum = self.weights[pos].sum()
            if row_sum > 0:
                self.weights[pos] /= row_sum
        
        self.temperature *= self.temp_decay
        self.update_dual_variables(solution)

    def improve_subproblem(self, solution: List[int], subproblem_indices: List[int]) -> Tuple[List[int], float]:
        """Uses sensitivity learning approach (KEEP THIS FROM PASTE 1)"""
        if all(i in range(len(solution)) for i in solution):
            problem_type = "TSP"
        else:
            problem_type = "Knapsack"
        
        improve_fn = create_problem_agnostic_improver(problem_type, self.problem_size)
        return improve_fn(self, solution, subproblem_indices)
        
    def improve_subproblem_pertubation(self, solution, subproblem_indices):
        """pertubation method"""
        current_solution = solution.copy()
        current_value = self.evaluate_fn(current_solution)
        
        for round in range(min(5, len(subproblem_indices))):
            # Single perturbation to estimate gradient
            perturbed = current_solution.copy()
            i, j = random.sample(subproblem_indices, min(2, len(subproblem_indices)))
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            
            perturbed_value = self.evaluate_fn(perturbed)
            gradient_estimate = (perturbed_value - current_value) / 1.0
            
            # Adaptive step based on gradient magnitude
            if abs(gradient_estimate) > 0.1:  # High gradient
                num_swaps = 1
            elif abs(gradient_estimate) < 0.01:  # Low gradient  
                num_swaps = min(3, len(subproblem_indices) // 2)
            else:
                num_swaps = 2
            
            # Apply multiple swaps based on adaptive step
            candidate = current_solution.copy()
            for _ in range(num_swaps):
                if len(subproblem_indices) >= 2:
                    k, l = random.sample(subproblem_indices, 2)
                    candidate[k], candidate[l] = candidate[l], candidate[k]
            
            # Evaluate and accept/reject
            candidate_value = self.evaluate_fn(candidate)
            if candidate_value > current_value:
                current_solution = candidate
                current_value = candidate_value
        
        return current_solution, current_value - self.evaluate_fn(solution)


    def construct_solution_with_decomposition(self) -> List[int]:
        """Construct a solution - PROBLEM-TYPE AWARE (not dataset-specific)"""
        
        # 1. CATEGORICAL / CONTINUOUS MODE (bounds-based)
        if hasattr(self, 'bounds') and self.bounds is not None:
            solution = []
            for pos in range(self.problem_size):
                bound = self.bounds[pos]
                if isinstance(bound, list):
                    # Categorical: select from discrete options using learned weights
                    val = self.select_action_for_position(pos, bound)
                    solution.append(val)
                elif isinstance(bound, tuple):
                    # Continuous: sample from range (could add learned bias later)
                    min_val, max_val = bound
                    solution.append(random.uniform(min_val, max_val))
                else:
                    solution.append(0)
            return solution
        
        # 2. BINARY MODE (knapsack-type)
        is_binary = hasattr(self, 'item_weights') and self.item_weights is not None
        
        if is_binary:
            solution = [0] * self.problem_size
            current_weight = 0.0
            
            for pos in range(self.problem_size):
                item_weight = self.item_weights[pos].item() if torch.is_tensor(self.item_weights[pos]) else self.item_weights[pos]
                
                if current_weight + item_weight <= self.knapsack_capacity:
                    # Use learned weights instead of static random!
                    val = self.select_action_for_position(pos, [0, 1])
                    solution[pos] = val
                    if val == 1:
                        current_weight += item_weight
                else:
                    solution[pos] = 0
            
            return solution
        
        # 3. PERMUTATION MODE (TSP-type) - default
        solution = [-1] * self.problem_size
        available = set(range(self.problem_size))
        
        for subproblem in self.subproblems:
            for pos in subproblem:
                if solution[pos] == -1:
                    value = self.select_action_for_position(pos, list(available))
                    solution[pos] = value
                    available.discard(value)
        
        # Fill any remaining positions
        remaining_positions = [i for i in range(self.problem_size) if solution[i] == -1]
        remaining_values = list(available)
        for pos, val in zip(remaining_positions, remaining_values):
            solution[pos] = val
        
        return solution

    def optimize(self, initial_solution: Optional[List[int]] = None) -> Tuple[List[int], float]:
        """Main optimization loop with global parameter sharing."""
        if initial_solution is None:
            initial_solution = self.construct_solution_with_decomposition()
        
        current_solution = initial_solution
        current_reward = self.evaluate_fn(current_solution)
        
        self.best_solution = current_solution.copy()
        self.best_reward = current_reward
        
        self.solutions.append(current_solution)
        self.rewards.append(current_reward)
        
        self.update_global_parameters(current_solution, current_reward)
        
        print(f"Starting Fixed Decomposed Bandit-UCB-Hedge...")
        print(f"Problem size: {self.problem_size}, Subproblems: {self.n_subproblems}")
        print(f"Initial reward: {current_reward:.4f}")
        print(f"Lagrangian: {self.use_lagrangian}, FTRL: {self.use_ftrl}")
        print(f"Accelerated Dual: {self.use_accelerated_dual}, Diminishing Overlap: {self.use_diminishing_overlap}")
        
        for iteration in range(self.max_iterations):
            iteration_start_reward = current_reward
            
            # NEW: Update overlap if using diminishing schedule
            self._update_overlap()
            
            for sp_idx, subproblem in enumerate(self.subproblems):
                improved_solution, improvement = self.improve_subproblem(current_solution, subproblem)
                
                if improvement > 0:
                    current_solution = improved_solution
                    current_reward = self.evaluate_fn(current_solution)
                    
                    self.update_global_parameters(current_solution, current_reward)
                    
                    if current_reward > self.best_reward:
                        self.best_solution = current_solution.copy()
                        self.best_reward = current_reward
                        print(f"Iteration {iteration}, SP {sp_idx}: New best = {self.best_reward:.4f}")
                        self.no_improvement_count = 0
            
            if iteration % 10 == 9:
                new_solution = self.construct_solution_with_decomposition()
                new_reward = self.evaluate_fn(new_solution)
                
                self.update_global_parameters(new_solution, new_reward)
                
                if new_reward > current_reward:
                    current_solution = new_solution
                    current_reward = new_reward
                    
                    if current_reward > self.best_reward:
                        self.best_solution = current_solution.copy()
                        self.best_reward = current_reward
                        print(f"Iteration {iteration}, Reconstruct: New best = {self.best_reward:.4f}")
                        self.no_improvement_count = 0
            
            if current_reward <= iteration_start_reward:
                self.no_improvement_count += 1
            else:
                self.no_improvement_count = 0
            
            self.solutions.append(current_solution)
            self.rewards.append(current_reward)
            
            if iteration % 10 == 0:
                avg_uncertainty = self.compute_uncertainty_for_positions(list(range(self.problem_size)))
                dual_norm = torch.norm(self.dual_vars).item() if self.use_lagrangian else 0.0
                ucb_conf = self.get_max_ucb_confidence_width()
                print(f"Iteration {iteration}/{self.max_iterations}, "
                      f"Current: {current_reward:.4f}, Best: {self.best_reward:.4f}, "
                      f"Uncertainty: {avg_uncertainty:.3f}, UCB_conf: {ucb_conf:.4f}, Temp: {self.temperature:.3f}, "
                      f"||Î»||: {dual_norm:.3f}, Overlap: {self.overlap}")
            
            if self.no_improvement_count >= self.patience:
                ucb_conf = self.get_max_ucb_confidence_width()
                if ucb_conf < 0.05:
                    print(f"Converged: UCB confidence {ucb_conf:.4f} < 0.05 (tight bounds)")
                else:
                    print(f"WARNING: Stopped but UCB confidence {ucb_conf:.4f} > 0.05 (may be premature)")
                break
        
        return self.best_solution, self.best_reward
    
    def get_statistics(self) -> Dict:
        """Get statistics about the optimization process."""
        stats = {
            'total_iterations': self.total_iterations,
            'best_reward': self.best_reward,
            'n_subproblems': self.n_subproblems,
            'final_temperature': self.temperature,
            'global_uncertainty': self.compute_uncertainty_for_positions(list(range(self.problem_size))),
            'avg_visits_per_position': self.visit_counts.sum().item() / self.problem_size,
            'convergence_iteration': len(self.rewards) - self.no_improvement_count,
            'dual_norm': torch.norm(self.dual_vars).item() if self.use_lagrangian else 0.0,
            'final_overlap': self.overlap  # NEW: Track final overlap
        }
        
        return stats


class AdvancedDecompositionWrapper:
    """
    Wrapper that ACTUALLY uses correlation, elite, and metric-based decomposition
    """
    
    def __init__(self, problem, **kwargs):
        self.problem = problem
        print(f"WRAPPER INIT RECEIVED: decomposition_size={kwargs.get('decomposition_size', 'NOT FOUND')}, overlap={kwargs.get('overlap', 'NOT FOUND')}")

        # Problem detection
        self.problem_name = problem.__class__.__name__
        self.problem_type = self._detect_problem_type()
        self.domain_type = self._detect_domain_type()
        self.n_objectives = self._get_num_objectives()
        self.problem_size = self._get_problem_size()
        
        # Extract wrapper-level parameters
        self.n_weight_vectors = kwargs.pop('n_weight_vectors', 15)
        self.base_iterations = kwargs.pop('max_iterations', 50)
        
        # Advanced decomposition control
        self.use_correlation = kwargs.pop('use_correlation_decomposition', False)
        self.use_elite = kwargs.pop('use_elite_decomposition', False)
        self.use_metric = kwargs.pop('use_metric_decomposition', True)
        self.correlation_threshold = kwargs.pop('correlation_threshold', 0.8) # 0.8 latest dec23: 0.6 | 0.5
        self.decomposition_update_freq = kwargs.pop('decomposition_update_freq', 80) # 20 (latest: dec23) 5
        
        # Remaining params for optimizer
        self.optimizer_params = kwargs
        
        # Initialize problem data
        self._initialize_problem_data()
        
        # Archives
        self.archive = []
        self.archive_objectives = []
        
        print(f"Advanced Decomposition Wrapper initialized")
        print(f"  Problem: {self.problem_type}, Size: {self.problem_size}")
        print(f"  Using: Correlation={self.use_correlation}, Elite={self.use_elite}, Metric={self.use_metric}")
        
    
    def _detect_problem_type(self):
        if 'TSP' in self.problem_name:
            return 'TSP'
        elif 'Knapsack' in self.problem_name:
            return 'Knapsack'
        elif 'CVRP' in self.problem_name:  # NEW
            return 'CVRP'
        return 'Unknown'
    
    def _detect_domain_type(self):
        if self.problem_type == 'TSP':
            return 'permutation'
        elif self.problem_type == 'Knapsack':
            return 'binary'
        elif self.problem_type == 'CVRP':  # NEW
            return 'permutation'  # CVRP uses permutation encoding
        return 'unknown'
    
    def _get_num_objectives(self):
        if hasattr(self.problem, 'num_objectives'):
            return self.problem.num_objectives
        elif hasattr(self.problem, '_n_objectives'):
            return self.problem._n_objectives
        return 2
    
    def _get_problem_size(self):
        if hasattr(self.problem, 'n_cities'):
            return self.problem.n_cities
        elif hasattr(self.problem, 'n_items'):
            return self.problem.n_items
        elif hasattr(self.problem, 'n_customers'):  # NEW
            return self.problem.n_customers
        return 100
    
    def _initialize_problem_data(self):
        if self.problem_type == 'TSP':
            self.distance_matrices = []
            for i in range(1, self.n_objectives + 1):
                if hasattr(self.problem, f'distances{i}'):
                    self.distance_matrices.append(
                        torch.tensor(getattr(self.problem, f'distances{i}'), dtype=torch.float32)
                    )
        elif self.problem_type == 'Knapsack':
            self.item_values = []
            for i in range(1, self.n_objectives + 1):
                if hasattr(self.problem, f'values{i}'):
                    self.item_values.append(
                        torch.tensor(getattr(self.problem, f'values{i}'), dtype=torch.float32)
                    )
            self.item_weights = torch.tensor(self.problem.weights, dtype=torch.float32)
            self.capacity = self.problem.capacity

        elif self.problem_type == 'CVRP':
            self.cvrp_distances = torch.tensor(self.problem.distances, dtype=torch.float32)
            self.cvrp_customers = self.problem.customers
            self.cvrp_vehicle_capacity = self.problem.vehicle_capacity
            self.cvrp_n_vehicles = self.problem.n_vehicles
    
    
    def _compute_objectives(self, solution):
        """Compute objectives with auto-fixing for invalid solutions"""
        if solution is None:
            if self.problem_type == 'Knapsack':
                return [-float('inf')] * self.n_objectives
            else:
                return [float('inf')] * self.n_objectives
        
        if self.problem_type == 'TSP':
            # Auto-fix invalid TSP solutions
            if len(solution) != self.problem_size or len(set(solution)) != self.problem_size:
                valid_solution = []
                used = set()
                
                for val in solution:
                    if isinstance(val, (int, np.integer)) and val not in used and 0 <= val < self.problem_size:
                        valid_solution.append(val)
                        used.add(val)
                
                for i in range(self.problem_size):
                    if i not in used:
                        valid_solution.append(i)
                
                solution = valid_solution
            
            objectives = []
            for dist_matrix in self.distance_matrices:
                if torch.is_tensor(dist_matrix):
                    dist_matrix = dist_matrix.cpu().numpy()
                
                total = sum(dist_matrix[solution[i]][solution[(i+1)%len(solution)]]
                          for i in range(len(solution)))
                objectives.append(total)
            
            return objectives
            
        elif self.problem_type == 'Knapsack':
            # Auto-fix non-binary solutions
            if not all(x in [0, 1] for x in solution):
                solution = [1 if x > 0.5 else 0 for x in solution]
            
            total_weight = sum(self.item_weights[i] * solution[i] for i in range(len(solution)))
            if total_weight > self.capacity:
                return [-float('inf')] * self.n_objectives
            
            objectives = []
            for values in self.item_values:
                total = sum(values[i] * solution[i] for i in range(len(solution)))
                objectives.append(total.item() if torch.is_tensor(total) else total)
            
            return objectives
        
        elif self.problem_type == 'CVRP':
            # CVRP: Convert permutation to routes and compute objectives
            # Solution is a permutation of customer indices (0 to n_customers-1)
            # which maps to customer IDs (1 to n_customers)
            
            # Auto-fix invalid permutation
            if len(solution) != self.problem_size or len(set(solution)) != self.problem_size:
                valid_solution = []
                used = set()
                for val in solution:
                    if isinstance(val, (int, np.integer)) and val not in used and 0 <= val < self.problem_size:
                        valid_solution.append(val)
                        used.add(val)
                for i in range(self.problem_size):
                    if i not in used:
                        valid_solution.append(i)
                solution = valid_solution
            
            # Convert permutation indices to customer IDs (add 1)
            customer_order = [s + 1 for s in solution]
            
            # Split into routes based on capacity
            routes = self._split_into_routes(customer_order)
            
            # Compute objectives using the problem's evaluate method
            return list(self.problem.evaluate(routes))

        return [float('inf')] * self.n_objectives
    
    
    def _split_into_routes(self, customer_order):
        """Split a customer ordering into routes respecting capacity constraints.
        Uses Worst-Fit bin packing: assign to route with most remaining capacity.
        This is a classic load-balancing heuristic (not CVRP-specific)."""
        
        # Use same number of vehicles as problem definition
        n_routes = self.cvrp_n_vehicles
        
        # Initialize routes
        routes = [[] for _ in range(n_routes)]
        route_demands = [0] * n_routes
        
        # Worst-Fit: assign each customer to the route with most remaining capacity
        for customer_id in customer_order:
            customer_demand = self.cvrp_customers[customer_id].demand
            
            # Find all routes that can fit this customer, with their current demands
            valid_routes = [(i, route_demands[i]) for i in range(n_routes) 
                          if route_demands[i] + customer_demand <= self.cvrp_vehicle_capacity]
            
            if valid_routes:
                # Worst-Fit: pick route with LOWEST demand (most remaining capacity)
                chosen = min(valid_routes, key=lambda x: x[1])[0]
                routes[chosen].append(customer_id)
                route_demands[chosen] += customer_demand
            else:
                # No valid route with capacity - find empty route
                empty_routes = [i for i in range(len(routes)) if not routes[i]]
                if empty_routes:
                    routes[empty_routes[0]].append(customer_id)
                    route_demands[empty_routes[0]] = customer_demand
                else:
                    # Overflow: create new route
                    routes.append([customer_id])
                    route_demands.append(customer_demand)
        
        # Remove empty routes
        routes = [r for r in routes if r]
        
        return routes


    def _create_evaluator(self, weights):
        def evaluate(solution):
            obj = self._compute_objectives(solution)
            if self.problem_type == 'Knapsack':
                if any(o == -float('inf') for o in obj):
                    return -float('inf')
                return sum(w * o for w, o in zip(weights, obj))
            else:
                if any(o == float('inf') for o in obj):
                    return -float('inf')
                return -sum(w * o for w, o in zip(weights, obj))
        return evaluate
    
    def _generate_initial_solution(self):
        if self.domain_type == 'permutation':
            solution = list(range(self.problem_size))
            random.shuffle(solution)
            return solution
        elif self.domain_type == 'binary':
            solution = [0] * self.problem_size
            if self.problem_type == 'Knapsack':
                current_weight = 0
                for i in range(self.problem_size):
                    item_weight = self.item_weights[i].item() if torch.is_tensor(self.item_weights[i]) else self.item_weights[i]
                    if current_weight + item_weight <= self.capacity:
                        if random.random() < 0.3:
                            solution[i] = 1
                            current_weight += item_weight
            return solution
        return None
    
    def _generate_weight_vectors(self):
        if self.n_objectives == 2:
            return [(i/(self.n_weight_vectors-1), 1-i/(self.n_weight_vectors-1)) 
                   for i in range(self.n_weight_vectors)]
        else:
            vectors = []
            for _ in range(self.n_weight_vectors):
                w = np.random.dirichlet(np.ones(self.n_objectives))
                vectors.append(tuple(w))
            return vectors
    
    def _update_archive(self, solution, objectives):
        dominated = False
        to_remove = []
        
        for i, arch_obj in enumerate(self.archive_objectives):
            if self.problem_type == 'Knapsack':
                if all(ao >= o for ao, o in zip(arch_obj, objectives)) and \
                   any(ao > o for ao, o in zip(arch_obj, objectives)):
                    dominated = True
                    break
                elif all(o >= ao for o, ao in zip(objectives, arch_obj)) and \
                     any(o > ao for o, ao in zip(objectives, arch_obj)):
                    to_remove.append(i)
            else:
                if all(ao <= o for ao, o in zip(arch_obj, objectives)) and \
                   any(ao < o for ao, o in zip(arch_obj, objectives)):
                    dominated = True
                    break
                elif all(o <= ao for o, ao in zip(objectives, arch_obj)) and \
                     any(o < ao for o, ao in zip(objectives, arch_obj)):
                    to_remove.append(i)
        
        if not dominated:
            for i in reversed(to_remove):
                del self.archive[i]
                del self.archive_objectives[i]
            self.archive.append(solution.copy())
            self.archive_objectives.append(objectives)
    
    def run(self):
        """Main optimization with advanced decomposition"""
        print(f"\nRunning Advanced Decomposition")
            
        weight_vectors = self._generate_weight_vectors()
        
        # Create custom optimizer class with advanced decomposition
        wrapper_self = self  # Capture reference for inner class
        
        class AdvancedOptimizer(FixedDecomposedBanditUCBHedge):
            def __init__(self, problem_size, evaluate_fn, **kwargs):
                # Store advanced features
                self.elite_solutions = []
                self.elite_values = []
                self.decomposition_counter = 0
                self.correlation_matrix = None
                self.metric_centers = []
                
                # Initialize base
                super().__init__(problem_size, evaluate_fn, **kwargs)
                
            def optimize(self, initial_solution=None):
                """Override optimize to add decomposition updates"""
                if initial_solution is None:
                    initial_solution = self.construct_solution_with_decomposition()
                
                current_solution = initial_solution
                current_reward = self.evaluate_fn(current_solution)
                
                self.best_solution = current_solution.copy()
                self.best_reward = current_reward
                
                self.solutions.append(current_solution)
                self.rewards.append(current_reward)
                self.update_global_parameters(current_solution, current_reward)
                
                for iteration in range(self.max_iterations):
                    # UPDATE DECOMPOSITION PERIODICALLY
                    if iteration % wrapper_self.decomposition_update_freq == 0 and iteration > 0:
                        self._update_decomposition()
                    
                    # Standard subproblem improvement
                    for sp_idx, subproblem in enumerate(self.subproblems):
                        improved_solution, improvement = self.improve_subproblem(
                            current_solution, subproblem)
                        
                        if improvement > 0:
                            current_solution = improved_solution
                            current_reward = self.evaluate_fn(current_solution)
                            
                            # Track elite
                            self._update_elite(current_solution, current_reward)
                            
                            self.update_global_parameters(current_solution, current_reward)
                            
                            if current_reward > self.best_reward:
                                self.best_solution = current_solution.copy()
                                self.best_reward = current_reward
                                self.no_improvement_count = 0
                    
                    # Periodic reconstruction
                    if iteration % 10 == 9:
                        new_solution = self.construct_solution_with_decomposition()
                        new_reward = self.evaluate_fn(new_solution)
                        
                        self._update_elite(new_solution, new_reward)
                        self.update_global_parameters(new_solution, new_reward)
                        
                        if new_reward > current_reward:
                            current_solution = new_solution
                            current_reward = new_reward
                            
                            if current_reward > self.best_reward:
                                self.best_solution = new_solution
                                self.best_reward = current_reward
                                self.no_improvement_count = 0
                    
                    self.solutions.append(current_solution)
                    self.rewards.append(current_reward)
                    
                    if current_reward <= self.best_reward:
                        self.no_improvement_count += 1
                    else:
                        self.no_improvement_count = 0
                    
                    if self.no_improvement_count >= self.patience:
                        break
                
                return self.best_solution, self.best_reward
            
            def _update_decomposition(self):
                """Create new decomposition using all strategies"""
                all_subproblems = []
                
                # Always keep some sliding windows
                sliding = self._create_simple_decomposition()
                all_subproblems.extend(sliding[:len(sliding)//3])
                
                # Add correlation-based if enabled and have data
                if wrapper_self.use_correlation and self.total_iterations > 20:
                    corr_groups = self._create_correlation_decomposition()
                    all_subproblems.extend(corr_groups)
                
                # Add elite-based if enabled and have elites
                if wrapper_self.use_elite and len(self.elite_solutions) >= 2:
                    elite_groups = self._create_elite_decomposition()
                    all_subproblems.extend(elite_groups)
                
                # Add metric-based if enabled
                if wrapper_self.use_metric:
                    metric_groups = self._create_metric_decomposition()
                    all_subproblems.extend(metric_groups)
                
                # Deduplicate
                unique = []
                seen = set()
                for sub in all_subproblems:
                    if len(sub) >= 2:
                        sig = tuple(sorted(sub[:min(5, len(sub))]))
                        if sig not in seen:
                            unique.append(sub)
                            seen.add(sig)
                
                if unique:
                    self.subproblems = unique
                    self.n_subproblems = len(unique)
                    print(f"    Updated to {self.n_subproblems} subproblems "
                          f"(corr:{len([s for s in unique if 'corr' in str(s)[:0]])})")
            
            def _create_correlation_decomposition(self):
                """Group by value correlation"""
                groups = []
                used = set()
                
                for i in range(self.problem_size):
                    if i in used or self.visit_counts[i].sum() < 10:
                        continue
                    
                    group = [i]
                    used.add(i)
                    
                    # Compute correlations with other positions
                    for j in range(self.problem_size):
                        if j not in used and self.visit_counts[j].sum() >= 10:
                            val_i = self.value_estimates[i]
                            val_j = self.value_estimates[j]
                            
                            if val_i.std() > 1e-6 and val_j.std() > 1e-6:
                                corr = torch.corrcoef(torch.stack([val_i, val_j]))[0,1].item()
                                if not np.isnan(corr) and corr > wrapper_self.correlation_threshold:
                                    group.append(j)
                                    used.add(j)
                                    if len(group) >= self.decomposition_size:
                                        break
                    
                    if len(group) >= 2:
                        groups.append(group)
                
                return groups[:5]
            
            def _create_elite_decomposition(self):
                """Group positions where elites differ"""
                groups = []
                
                for i in range(len(self.elite_solutions)-1):
                    diff_pos = [p for p in range(min(len(self.elite_solutions[i]), 
                                                     len(self.elite_solutions[i+1])))
                               if self.elite_solutions[i][p] != self.elite_solutions[i+1][p]]
                    
                    if len(diff_pos) >= 2:
                        for start in range(0, len(diff_pos), self.decomposition_size):
                            group = diff_pos[start:start+self.decomposition_size]
                            if len(group) >= 2:
                                groups.append(group)
                
                return groups[:3]
            
            def _create_metric_decomposition(self):
                """Create metric-based groups"""
                groups = []
                
                # For TSP: use geometric clustering
                if wrapper_self.problem_type == 'TSP' and hasattr(wrapper_self, 'distance_matrices'):
                    # Select random centers
                    n_centers = min(5, self.problem_size // 10)
                    centers = random.sample(range(self.problem_size), n_centers)
                    
                    for center in centers:
                        # Get nearest neighbors from first distance matrix
                        dist_mat = wrapper_self.distance_matrices[0]
                        if torch.is_tensor(dist_mat):
                            distances = [(dist_mat[center][i].item(), i) 
                                       for i in range(self.problem_size)]
                        else:
                            distances = [(dist_mat[center][i], i) 
                                       for i in range(self.problem_size)]
                        distances.sort()
                        
                        group = [idx for _, idx in distances[:self.decomposition_size]]
                        if len(group) >= 2:
                            groups.append(group)
                
                # For CVRP: use geometric clustering based on customer distances
                elif wrapper_self.problem_type == 'CVRP' and hasattr(wrapper_self, 'cvrp_distances'):
                    n_centers = min(5, self.problem_size // 10)
                    if n_centers < 1:
                        n_centers = 1
                    centers = random.sample(range(self.problem_size), n_centers)
                    
                    for center in centers:
                        # Customer indices are 0 to n_customers-1 in our permutation
                        # But distance matrix uses 1 to n_customers (plus depot at 0)
                        # So we need to offset by 1
                        dist_mat = wrapper_self.cvrp_distances
                        if torch.is_tensor(dist_mat):
                            # center+1 because customer IDs are 1-indexed in distance matrix
                            distances = [(dist_mat[center+1][i+1].item(), i) 
                                       for i in range(self.problem_size)]
                        else:
                            distances = [(dist_mat[center+1][i+1], i) 
                                       for i in range(self.problem_size)]
                        distances.sort()
                        
                        group = [idx for _, idx in distances[:self.decomposition_size]]
                        if len(group) >= 2:
                            groups.append(group)

                # For binary: contiguous chunks
                else:
                    for _ in range(3):
                        start = random.randint(0, max(0, self.problem_size - self.decomposition_size))
                        group = list(range(start, min(start + self.decomposition_size, self.problem_size)))
                        if len(group) >= 2:
                            groups.append(group)
                
                return groups[:3]
            
            def _update_elite(self, solution, value):
                """Maintain elite solutions"""
                if len(self.elite_solutions) < 5:
                    self.elite_solutions.append(solution.copy())
                    self.elite_values.append(value)
                else:
                    min_idx = np.argmin(self.elite_values)
                    if value > self.elite_values[min_idx]:
                        self.elite_solutions[min_idx] = solution.copy()
                        self.elite_values[min_idx] = value
        
        # Run with each weight vector
        for idx, weights in enumerate(weight_vectors):
            weight_str = ", ".join(f"{w:.2f}" for w in weights)
            print(f"\nWeight {idx+1}/{len(weight_vectors)}: ({weight_str})")
            
            evaluate_fn = self._create_evaluator(weights)
            
            # Scale parameters with problem size
            scale_factor = np.sqrt(self.problem_size / 20)
            
            params = self.optimizer_params.copy()
            params['max_iterations'] = int(self.base_iterations * scale_factor)
            
            # Problem-specific data
            if self.problem_type == 'Knapsack':
                params['item_weights'] = self.item_weights
                params['knapsack_capacity'] = self.capacity
            
            # Create advanced optimizer
            optimizer = AdvancedOptimizer(
                problem_size=self.problem_size,
                evaluate_fn=evaluate_fn,
                **params
            )
            
            initial = self._generate_initial_solution()
            best_sol, _ = optimizer.optimize(initial_solution=initial)
            
            # Collect solutions
            for sol in optimizer.solutions[::5]:
                obj = self._compute_objectives(sol)
                if self.problem_type == 'Knapsack':
                    if all(o != -float('inf') for o in obj):
                        self._update_archive(sol, obj)
                else:
                    if all(o != float('inf') for o in obj):
                        self._update_archive(sol, obj)
            
            print(f"  Archive: {len(self.archive)} solutions")
        
        result = [[sol, obj] for sol, obj in zip(self.archive, self.archive_objectives)]
        print(f"\nFinal Pareto front: {len(result)} solutions")
        
        return result if result else [self._generate_initial_solution(), 
                                      self._compute_objectives(self._generate_initial_solution())]


class CachedAdvancedBiKPWrapper(AdvancedDecompositionWrapper):
    """Extends AdvancedDecompositionWrapper with caching for BiKP"""
    
    def __init__(self, problem, **kwargs):
        # Initialize cache before parent init
        self.evaluation_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Call parent init
        super().__init__(problem, **kwargs)
        
        print(f"Added caching to AdvancedDecompositionWrapper")
    
    def _compute_objectives(self, solution):
        """Override to use caching"""
        cache_key = tuple(solution)
        
        if cache_key in self.evaluation_cache:
            self.cache_hits += 1
            return self.evaluation_cache[cache_key]
        else:
            self.cache_misses += 1
            
            # Call parent's compute_objectives
            result = super()._compute_objectives(solution)
            
            self.evaluation_cache[cache_key] = result
            return result
    
    def run(self):
        """Run parent's optimization then print cache stats"""
        result = super().run()
        
        # Print cache statistics
        total_calls = self.cache_hits + self.cache_misses
        if total_calls > 0:
            hit_rate = self.cache_hits / total_calls * 100
            print(f"\nCache Statistics:")
            print(f"  Hit rate: {hit_rate:.1f}%")
            print(f"  Evaluations saved: {self.cache_hits}")
        
        return result