#from Algorithms.dcpso import DCPSO
from Algorithms.mcpso import MCPSO
from cec2017.functions import f4
from benchmarks.cec2017 import rosenbrock
from benchmarks.benchmarks import standard_rosenbrock
from Policies.interval_based_split_policy import IntervalBasedSplitPolicy
from Policies.stagnation_based_split_policy import StagnationBasedSplitPolicy
from Policies.diversity_based_merge_policy import DiversityBasedMergePolicy
from Policies.random_merge_policy import RandomMergePolicy
from Policies.interval_based_merge_policy import IntervalBasedMergePolicy
from Policies.repeating_interval_based_policy import RepeatingIntervalBasedPolicy
import numpy as np

#np.random.seed(42)
DIMS = 4
results = []
for run in range(1):
    mcpso = MCPSO(
        swarm_size = 100,
        dims = DIMS,
        upper_bounds = [1000]*DIMS,
        lower_bounds = [-1000]*DIMS,
        obj_function = standard_rosenbrock,
        num_iterations = 3000,
        neighborhood_size = 3,
        merge_policy= IntervalBasedMergePolicy(
            merge_factor=3,
        )
        # merge_policy = DiversityBasedMergePolicy(
        #     patience=5,
        #     stagnation_threshold = 1e-8,
        #     merge_on_high_diversity = False,
        # )
        # merge_policy = RandomMergePolicy(
        #         patience=5,
        #         stagnation_threshold = 1e-8,
        #       )
        # merge_policy= RepeatingIntervalBasedPolicy(
        #     split_factor=10,
        #     merge_factor=10,
        #     num_phases=5,
        #     starting_phase='merge'
        # )
    )

    mcpso.init()
    result, d =mcpso.run()
    results.append(result)
    print(d)
