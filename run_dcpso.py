#imports from python files
from Algorithms.swarm import Swarm
from Algorithms.dcpso import DCPSO
from cec2017.functions import *
from benchmarks.cec2017 import *
from benchmarks.benchmarks import *
from Policies.interval_based_split_policy import IntervalBasedSplitPolicy
from Policies.stagnation_based_split_policy import StagnationBasedSplitPolicy
from Policies.repeating_interval_based_policy import RepeatingIntervalBasedPolicy
from Policies.interaction_based_policy import InteractionBasedPolicy
import numpy as np

#np.random.seed(32)
DIMS = 100

results = []
for run in range(1):
    dcpso = DCPSO(
        swarm_size = 100,
        dims = DIMS,
        upper_bounds = [1000]*DIMS,
        lower_bounds = [-1000]*DIMS,
        obj_function = standard_rosenbrock,
        num_iterations = 3000,
        neighborhood_size = 3,
        split_policy = StagnationBasedSplitPolicy(
                patience = 2,
                stagnation_threshold =1e-2,
                split_factor = 3,
        )
    #     split_policy = IntervalBasedSplitPolicy(
    #         split_factor = 2
    #   )
        # split_policy= RepeatingIntervalBasedPolicy(
        #     split_factor=2,
        #     merge_factor=2,
        #     num_phases=10,
        #     starting_phase='split'
        # )
    #     split_policy = InteractionBasedPolicy(
    #         split_factor = 2
    #   )
    )

    dcpso.init()
    result, cv, data  = dcpso.run()
    results.append(result)
    print(cv)

print(data)

