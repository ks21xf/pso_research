#imports from python files
from Algorithms import dcpso,mcpso
from cec2017.functions import all_functions


from Policies.interval_based_split_policy import IntervalBasedSplitPolicy
from Policies.stagnation_based_split_policy import StagnationBasedSplitPolicy
from Policies.repeating_interval_based_policy import RepeatingIntervalBasedPolicy
from Policies.interaction_based_policy import InteractionBasedPolicy
from Policies.diversity_based_merge_policy import DiversityBasedMergePolicy
from Policies.random_merge_policy import RandomMergePolicy
from Policies.interval_based_merge_policy import IntervalBasedMergePolicy
from Policies.interval_based_split_policy import IntervalBasedSplitPolicy

import numpy as np


DIMS = 50
NUM_RUNS = 3
MAIN_SEED = 11111
phases = [2,5,10,20,40,60,80,100,200,300,400,600]


print(f"ALGORITHM: repeatingintervalbasedpolicy")
for phase in phases:
    results = []
    print(f"NUM_PHASES: {phase}")
    print(f"RUNS: [")
    for run in range(NUM_RUNS):

                seed = MAIN_SEED + run #method to get a unique seed for each run
                np.random.seed(seed)

                pso = dcpso.DCPSO(
                    swarm_size = 100,
                    dims = DIMS,
                    upper_bounds = [100]*DIMS,
                    lower_bounds = [-100]*DIMS,
                    obj_function = all_functions[0],
                    num_iterations = 3000,
                    neighborhood_size = 3,
                    split_policy = RepeatingIntervalBasedPolicy(
                        split_factor=2,
                        merge_factor=2,
                        num_phases=phase,
                        starting_phase='split'
                    ),
                    verbose=False
                )
                pso.init()
                result, *_ = pso.run()
                print(f"{result[0]}")
                results.append(result)
    print("]")
    print(f"MEAN:{sum(results)/len(results)}")


