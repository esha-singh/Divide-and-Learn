
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import logging
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional, Union
from MOCO.problems import BiObjectiveTSP, MultiObjectiveKnapsack
from MOCO.evaluation import MOCOEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NSGA2")



def evaluate_nsga2_with_evaluator(num_runs=5):
    """Evaluate NSGA-II using MOCOEvaluator"""
    # Common parameters
    nsga2_params = {
        'population_size': 100,
        'n_generations': 100,#200,
        'crossover_prob': 0.9,
        'mutation_prob': 0.1,
        'tournament_size':3,
        'verbose': True
    }
    
    
    # Problem parameters
    tsp_problem_params = {'n_cities': 20}
    knapsack_problem_params = {'n_items': 200, 'n_objectives': 2, 'capacity': 25.0}
    
    # Initialize evaluator with reference points
    # For TSP (minimization), use a point above the Pareto front
    # tsp_evaluator = MOCOEvaluator(reference_point=(2494, 2402), confidence_level=0.95)
    tsp_evaluator = MOCOEvaluator(reference_point=(20,20), confidence_level=0.95)
    
    # For Knapsack (maximization), use a point below the Pareto front
    # Use (0, 0) as reference point for maximization
    # knapsack_evaluator = MOCOEvaluator(reference_point=(-7.85, -8.99), confidence_level=0.95)
    knapsack_evaluator = MOCOEvaluator(reference_point=(5, 5), confidence_level=0.95)
    
    print("\n" + "="*50)
    print(f"Evaluating NSGA-II with {num_runs} runs per problem:")
    print("="*50)
    
    # Evaluate on BiObjectiveTSP
    # print("\nEvaluating on BiObjectiveTSP:")
    # tsp_result = tsp_evaluator.evaluate_algorithm(
    #     algorithm_class=NSGA2,
    #     problem_class=BiObjectiveTSP,
    #     algorithm_name="NSGA-II",
    #     parameters=nsga2_params,
    #     problem_params=tsp_problem_params,
    #     num_runs=num_runs
    # )
    
    # Evaluate on MultiObjectiveKnapsack
    print("\nEvaluating on MultiObjectiveKnapsack:")
    knapsack_result = knapsack_evaluator.evaluate_algorithm(
        algorithm_class=NSGA2,
        problem_class=MultiObjectiveKnapsack,
        algorithm_name="NSGA-II",
        parameters=nsga2_params,
        problem_params=knapsack_problem_params,
        num_runs=num_runs
    )
    
    # Generate reports
    # print("\nGenerating TSP report:")
    # tsp_evaluator.generate_report()
    
    print("\nGenerating Knapsack report:")
    knapsack_evaluator.generate_report()
    
    # Print results
    print("\nAggregated Results:")
    
    # print("\nTSP Results:")
    # for result in tsp_evaluator.results:
    #     print(f"\n{result.algorithm_name} on {result.problem_name}:")
    #     print(f"Average Runtime: {result.runtime:.2f} seconds")
    #     print(f"Final Hypervolume: {result.hypervolume:.4f}")
    #     print(f"Final Non-dominated solutions: {result.num_nondominated}")
    
    print("\nKnapsack Results:")
    for result in knapsack_evaluator.results:
        print(f"\n{result.algorithm_name} on {result.problem_name}:")
        print(f"Average Runtime: {result.runtime:.2f} seconds")
        print(f"Final Hypervolume: {result.hypervolume:.4f}")
        print(f"Final Non-dominated solutions: {result.num_nondominated}")
    
    # Plot comparison
    try:
        # print("\nGenerating TSP visualizations...")
        # tsp_evaluator.plot_comparison()
        # tsp_evaluator.plot_pareto_front(show_all=True)
        
        print("\nGenerating Knapsack visualizations...")
        knapsack_evaluator.plot_comparison()
        knapsack_evaluator.plot_pareto_front(show_all=True)
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    return tsp_evaluator, knapsack_evaluator


def test_nsga2_on_tsp(n_cities=20, population_size=100, n_generations=200):
    """Test NSGA-II on BiObjectiveTSP"""
    logger.info(f"Testing NSGA-II on BiObjectiveTSP with {n_cities} cities")
    
    # Create problem instance
    problem = BiObjectiveTSP(n_cities=n_cities)
    
    # Create and run optimizer
    optimizer = NSGA2(
        problem=problem,
        population_size=population_size,
        n_generations=n_generations,
        crossover_prob=0.9,
        mutation_prob=0.1,
        verbose=True
    )
    
    # Run optimization
    pareto_front = optimizer.run()
    
    # Plot results
    optimizer.plot_convergence()
    optimizer.plot_pareto_front()
    
    return optimizer


def test_nsga2_on_knapsack(n_items=50, n_objectives=2, population_size=100, n_generations=200):
    """Test NSGA-II on MultiObjectiveKnapsack"""
    logger.info(f"Testing NSGA-II on MultiObjectiveKnapsack with {n_items} items")
    
    # Create problem instance
    problem = MultiObjectiveKnapsack(n_items=n_items, n_objectives=n_objectives, capacity=10.0)
    
    # Create and run optimizer
    optimizer = NSGA2(
        problem=problem,
        population_size=population_size,
        n_generations=n_generations,
        crossover_prob=0.9,
        mutation_prob=0.1,
        verbose=True
    )
    
    # Run optimization
    pareto_front = optimizer.run()
    
    # Plot results
    optimizer.plot_convergence()
    if n_objectives == 2:
        optimizer.plot_pareto_front()
    
    return optimizer



if __name__ == "__main__":
    # Run a single test on BiObjectiveTSP
    # print("\nRunning NSGA-II on BiObjectiveTSP:")
    # tsp_optimizer = test_nsga2_on_tsp(n_cities=20, population_size=100, n_generations=200)
    
    # Run a single test on MultiObjectiveKnapsack
    print("\nRunning NSGA-II on MultiObjectiveKnapsack:")
    # knapsack_optimizer = test_nsga2_on_knapsack(n_items=50, n_objectives=2, population_size=100, n_generations=200)
    
    # Run evaluation with MOCOEvaluator
    # print("\nEvaluating NSGA-II with MOCOEvaluator:")
    tsp_eval, knapsack_eval = evaluate_nsga2_with_evaluator(num_runs=2)