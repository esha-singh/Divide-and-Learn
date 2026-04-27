import os
import yaml
import time
import numpy as np
from typing import Dict, Any, List

# Import necessary MOCO components
from MOCO.problems import (
    BiObjectiveTSP, 
    MultiObjectiveKnapsack,
    TriObjectiveTSP
)
from MOCO.evaluation import MOCOEvaluator

def run_algorithm_with_timing(algorithm_class, problem_class, num_runs=30, **kwargs):
    """
    Run an algorithm multiple times with detailed timing tracking.
    
    Parameters:
    - algorithm_class: The algorithm class to instantiate
    - problem_class: The problem class to solve
    - num_runs: Number of independent runs (default 30)
    - **kwargs: Additional parameters for problem and algorithm instantiation
    
    Returns:
    - A dictionary containing detailed timing information and results
    """
    # Initialize timing and results storage
    run_times = []
    solutions_list = []
    
    # Extract problem parameters
    problem_params = kwargs.get('problem_params', {})
    algorithm_params = kwargs.get('algorithm_params', {})
    
    # Perform multiple runs
    for run in range(num_runs):
        # Create problem instance
        problem = problem_class(**problem_params)
        
        # Create algorithm instance - pass the problem here
        algorithm = algorithm_class(problem=problem, **algorithm_params)
        
        # Start timing
        start_time = time.time()
        
        # Run the algorithm with run() method without passing problem
        try:
            solutions = algorithm.run()  # WSLKH and WSDP use run() without arguments
            solutions_list.append(solutions)
        except Exception as e:
            print(f"Error in run {run+1}: {e}")
            solutions_list.append(None)
        
        # Stop timing
        end_time = time.time()
        
        # Calculate and store run time
        run_time = end_time - start_time
        run_times.append(run_time)
    
    # Calculate timing statistics
    timing_stats = {
        'total_time': sum(run_times),
        'mean_time': np.mean(run_times),
        'std_time': np.std(run_times),
        'min_time': np.min(run_times),
        'max_time': np.max(run_times),
        'runs': num_runs
    }
    
    return {
        'timing_stats': timing_stats,
        'solutions': solutions_list,
        'run_times': run_times
    }

def print_timing_details(timing_results, algorithm_name, problem_name, size):
    """
    Print detailed timing information for an algorithm run.
    
    Parameters:
    - timing_results: Dictionary containing timing information
    - algorithm_name: Name of the algorithm
    - problem_name: Name of the problem
    - size: Problem size category
    """
    stats = timing_results['timing_stats']
    
    print(f"\nTiming Details for {algorithm_name} on {problem_name} ({size}):")
    print(f"Number of Runs: {stats['runs']}")
    print(f"Total Execution Time: {stats['total_time']:.4f} seconds")
    print(f"Mean Execution Time per Run: {stats['mean_time']:.4f} seconds")
    print(f"Standard Deviation of Run Times: {stats['std_time']:.4f} seconds")
    print(f"Minimum Run Time: {stats['min_time']:.4f} seconds")
    print(f"Maximum Run Time: {stats['max_time']:.4f} seconds")
    
    # Percentile calculations
    run_times = timing_results['run_times']
    percentiles = [25, 50, 75, 90]
    for p in percentiles:
        print(f"{p}th Percentile Run Time: {np.percentile(run_times, p):.4f} seconds")

def benchmark_problem_specific():
    """
    Benchmark function for problem-specific algorithms (WS_LKH for TSP, WS_DP for Knapsack)
    """
    # Import algorithms
    from WS_LKH_DP import WSLKH, WSDP
    
    # Generate or load configuration
    config_path = 'configs/benchmark_ws_config.yaml'
    if not os.path.exists(config_path):
        generate_ws_config()
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Mapping of problems to specific algorithms
    problem_algorithm_map = {
        'BiObjectiveTSP': {'class': BiObjectiveTSP, 'algorithm': WSLKH, 'ref_type': 'BiTSP'},
        'TriObjectiveTSP': {'class': TriObjectiveTSP, 'algorithm': WSLKH, 'ref_type': 'TriTSP'},
        'MultiObjectiveKnapsack': {'class': MultiObjectiveKnapsack, 'algorithm': WSDP, 'ref_type': 'BiKP'}
    }
    
    # Prepare algorithms dictionary
    algorithms = {}
    algorithm_params = {}
    
    # Register algorithms from config
    for problem_config in config['benchmark'].get('problems', []):
        problem_name = problem_config['name']
        
        # Skip if problem not supported
        if problem_name not in problem_algorithm_map:
            print(f"Skipping {problem_name}: Not supported")
            continue
        
        # Find algorithm in config
        algo_config = None
        for algo in problem_config.get('algorithms', []):
            if 'weighted_sum' in algo['name'].lower() or 'ws' in algo['name'].lower():
                algo_config = algo
                break
        
        if not algo_config:
            print(f"No weighted sum algorithm found for {problem_name}")
            continue
        
        # Register algorithm
        algo_name = f"ws_{problem_name}"
        algorithms[algo_name] = problem_algorithm_map[problem_name]['algorithm']
        algorithm_params[algo_name] = algo_config.get('parameters', {})
        
        print(f"Registered {algo_name} using {algorithms[algo_name].__name__}")
    
    if not algorithms:
        print("No algorithms registered!")
        return None
    
    # Run benchmark
    print(f"\nRunning benchmark with {len(algorithms)} algorithms")
    
    # Define problem sizes to benchmark
    sizes = ['small', 'medium']
    
    # Problem configurations based on new problem classes
    problem_configs = {
        'BiObjectiveTSP': {
            'small': {'n_cities': 20},
            'medium': {'n_cities': 50}
        },
        'TriObjectiveTSP': {
            'small': {'n_cities': 20},
            'medium': {'n_cities': 50}
        },
        'MultiObjectiveKnapsack': {
            'small': {'n_items': 50, 'n_objectives': 2, 'capacity': 10.0},
            'medium': {'n_items': 100, 'n_objectives': 3, 'capacity': 20.0}
        }
    }
    
    # Create a master evaluator for getting standard reference points
    master_evaluator = MOCOEvaluator(reference_point=(10, 10))
    
    # Store results
    results = {}
    
    # Evaluation section
    for problem_name, problem_info in problem_algorithm_map.items():
        if problem_name not in results:
            results[problem_name] = {}
        
        for size in sizes:
            # Skip if problem config not available
            if size not in problem_configs.get(problem_name, {}):
                print(f"Skipping {problem_name} {size}: No configuration")
                continue
            
            # Get problem parameters
            problem_params = problem_configs[problem_name][size]
            
            # Determine reference point type and size
            if problem_name == 'BiObjectiveTSP':
                ref_type = 'BiTSP'
                ref_size = problem_params['n_cities']
            elif problem_name == 'TriObjectiveTSP':
                ref_type = 'TriTSP'
                ref_size = problem_params['n_cities']
            elif problem_name == 'MultiObjectiveKnapsack':
                # Adjust reference point type based on number of objectives
                n_objectives = problem_params.get('n_objectives', 2)
                ref_type = 'BiKP' if n_objectives == 2 else 'TriKP'
                ref_size = problem_params['n_items']
            else:
                ref_type = problem_info['ref_type']
                ref_size = problem_params.get('n_cities', problem_params.get('n_items', 0))
            
            # Get standard reference point
            try:
                standard_ref_point = master_evaluator.get_standard_points(
                    problem_type=ref_type,
                    problem_size=ref_size
                )
                reference_point = standard_ref_point.get('reference')
                
                # If using TriKP but reference point not defined, create a large one
                if ref_type == 'TriKP' and (reference_point is None or len(reference_point) != 3):
                    reference_point = (500, 500, 500)
            except Exception as e:
                print(f"Could not get standard reference point: {e}")
                # Create default reference point with appropriate dimensions
                if problem_name == 'TriObjectiveTSP':
                    reference_point = (35, 35, 35)
                elif problem_name == 'MultiObjectiveKnapsack' and problem_params.get('n_objectives', 2) == 3:
                    reference_point = (500, 500, 500)
                else:
                    reference_point = tuple([10.0] * problem_params.get('n_objectives', 2))
            
            # Create evaluator with the reference point
            evaluator = MOCOEvaluator(
                reference_point=reference_point,
                results_dir="benchmark_results"
            )
            
            # Print info
            print(f"\nBenchmarking {problem_name} ({size}):")
            print(f"Problem Parameters: {problem_params}")
            print(f"Reference Point: {reference_point}")
            
            # Get compatible algorithm
            algo_name = f"ws_{problem_name}"
            if algo_name not in algorithms:
                print(f"No algorithm registered for {problem_name}")
                continue
            
            # Get algorithm class and parameters
            algorithm_class = algorithms[algo_name]
            params = algorithm_params.get(algo_name, {})
            
            try:
                # Run algorithm with timing
                timing_results = run_algorithm_with_timing(
                    algorithm_class=algorithm_class,
                    problem_class=problem_info['class'],
                    problem_params=problem_params,
                    algorithm_params=params,
                    num_runs=config['benchmark'].get('num_runs', 5)  # Default to 5 runs
                )
                
                # Print detailed timing information
                print_timing_details(
                    timing_results, 
                    algorithm_class.__name__, 
                    problem_name, 
                    size
                )

                # Run evaluation
                result = evaluator.evaluate_algorithm(
                    algorithm_class=algorithm_class,
                    problem_class=problem_info['class'],
                    algorithm_name=algo_name,
                    parameters=params,
                    problem_params=problem_params,
                    num_runs=config['benchmark'].get('num_runs', 1)
                )
                
                # Add timing results to the result object
                result.timing_details = timing_results['timing_stats']

                # Get solutions from the timing results for hypervolume calculation
                all_solutions = timing_results['solutions']
                valid_solutions = [sol for sol in all_solutions if sol is not None]
                combined_solutions = []
                for sol_list in valid_solutions:
                    if sol_list:
                        combined_solutions.extend(sol_list)
                        
                # Get non-dominated solutions if the evaluator has this method
                if hasattr(evaluator, '_get_nondominated') and combined_solutions:
                    nondom_solutions = evaluator._get_nondominated(combined_solutions)
                    
                    # If evaluator has method to calculate hypervolume metrics
                    if hasattr(evaluator, '_calculate_hypervolume_metrics'):
                        raw_hv_metrics = evaluator._calculate_hypervolume_metrics(nondom_solutions, ref_type, ref_size)
                        result.raw_hv_metrics = raw_hv_metrics

                # Get reference and ideal points for manual normalization
                reference_point = evaluator.reference_point
                ideal_point = None

                # Try to get ideal point from standard points
                try:
                    standard_points = evaluator.get_standard_points(ref_type, ref_size)
                    if standard_points and 'ideal' in standard_points:
                        ideal_point = standard_points['ideal']
                except:
                    pass
                    
                # If ideal_point is still None, create a default one
                if ideal_point is None:
                    # Default ideal point with appropriate dimensionality
                    ideal_point = tuple([0] * len(reference_point))

                # Store reference and ideal points for normalization
                result.reference_point = reference_point
                result.ideal_point = ideal_point

                # Calculate custom normalized hypervolume
                ref_volume = 1.0
                for i in range(len(reference_point)):
                    ref_volume *= (reference_point[i] - ideal_point[i])

                if ref_volume > 0 and hasattr(result, 'hypervolume'):
                    result.normalized_hypervolume = result.hypervolume / ref_volume
                else:
                    result.normalized_hypervolume = 0.0

                # Display results
                print(f"\n{algo_name} Results:")
                print(f"Runtime: {result.runtime:.4f} seconds")
                if hasattr(result, 'hypervolume'):
                    print(f"Raw Hypervolume: {result.hypervolume:.4f} (using {result.ideal_point} ideal point)")
                print(f"Reference Point: {reference_point}")
                print(f"Ideal Point: {ideal_point}")
                print(f"Reference Volume: {ref_volume:.1f}")
                if hasattr(result, 'normalized_hypervolume'):
                    print(f"Normalized Hypervolume: {result.normalized_hypervolume:.4f}")
                if hasattr(result, 'num_nondominated'):
                    print(f"Non-dominated Solutions: {result.num_nondominated}")
                
                # Store result
                if size not in results[problem_name]:
                    results[problem_name][size] = []
                results[problem_name][size].append(result)
                
            except Exception as e:
                print(f"Error benchmarking {algo_name} on {problem_name} ({size}): {e}")
                import traceback
                traceback.print_exc()
    
    # Analyze results
    analyze_benchmark_results(results)
    
    return results

def generate_ws_config():
    """Generate a simple configuration file for weighted sum algorithms"""
    os.makedirs('configs', exist_ok=True)
    
    example_config = {
        'benchmark': {
            'num_runs': 5,
            'problems': [
                {
                    'name': 'BiObjectiveTSP',
                    'algorithms': [
                        {
                            'name': 'weighted_sum_lkh',
                            'parameters': {
                                'num_weights': 40
                            }
                        }
                    ]
                },
                {
                    'name': 'TriObjectiveTSP',
                    'algorithms': [
                        {
                            'name': 'weighted_sum_lkh',
                            'parameters': {
                                'num_weights': 40
                            }
                        }
                    ]
                },
                {
                    'name': 'MultiObjectiveKnapsack',
                    'algorithms': [
                        {
                            'name': 'weighted_sum_dp',
                            'parameters': {
                                'num_weights': 40
                            }
                        }
                    ]
                }
            ]
        }
    }
    
    with open('configs/benchmark_ws_config.yaml', 'w') as file:
        yaml.dump(example_config, file, default_flow_style=False)
    
    print("Configuration generated at 'configs/benchmark_ws_config.yaml'")

def analyze_benchmark_results(results):
    """
    Enhanced analysis of benchmark results that displays multiple hypervolume metrics
    """
    if not results:
        print("\nNo results to analyze")
        return
        
    print("\nBenchmark Result Analysis:")
    for problem_type, problem_results in results.items():
        print(f"\n{problem_type} Performance:")
        for size, size_results in problem_results.items():
            if not size_results:
                print(f"  {size.capitalize()} Size: No results")
                continue
                
            print(f"\n  {size.capitalize()} Size:")
            for result in size_results:
                print(f"    {result.algorithm_name}:")
                if hasattr(result, 'timing_details'):
                    print(f"      Total Runtime: {result.timing_details['total_time']:.4f} seconds")
                    print(f"      Mean Runtime: {result.timing_details['mean_time']:.4f} seconds")
                
                # Display the raw hypervolume (as stored in the result object)
                if hasattr(result, 'hypervolume'):
                    print(f"      Raw Hypervolume: {result.hypervolume:.4f}")
                
                # Display normalized hypervolume
                if hasattr(result, 'normalized_hypervolume'):
                    print(f"      Normalized Hypervolume: {result.normalized_hypervolume:.4f}")
                
                # Display number of non-dominated solutions
                if hasattr(result, 'num_nondominated'):
                    print(f"      Non-dominated Solutions: {result.num_nondominated}")
                    
                # Print additional metrics if available
                if hasattr(result, 'statistics') and result.statistics:
                    if 'spread' in result.statistics:
                        spread = result.statistics['spread'].get('mean', 0.0)
                        print(f"      Solution Spread: {spread:.4f}")
                    
                    if 'reliability_scores' in result.statistics:
                        reliability = result.statistics['reliability_scores'].get('hypervolume', 0.0)
                        print(f"      Reliability Score: {reliability:.1f}/100")

if __name__ == "__main__":
    benchmark_problem_specific()