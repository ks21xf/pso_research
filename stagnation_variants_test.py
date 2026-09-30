#imports from python files
from Algorithms import dcpso,mcpso
from cec2017.functions import *
from benchmarks.benchmarks import *
from benchmarks.cec2017 import *

from Policies.interval_based_split_policy import IntervalBasedSplitPolicy
from Policies.stagnation_based_split_policy import StagnationBasedSplitPolicy
from Policies.repeating_interval_based_policy import RepeatingIntervalBasedPolicy
from Policies.interaction_based_policy import InteractionBasedPolicy
from Policies.diversity_based_merge_policy import DiversityBasedMergePolicy
from Policies.random_merge_policy import RandomMergePolicy
from Policies.interval_based_merge_policy import IntervalBasedMergePolicy
from Policies.interval_based_split_policy import IntervalBasedSplitPolicy

import numpy as np
import copy
import argparse

#add and parse args
parser = argparse.ArgumentParser(description="A script for running tests on stagnation variants.")
parser.add_argument("-algo_name", choices=["stagnationbasedsplitpolicy", "randommergepolicy", "highdiversitybasedmergepolicy","lowdiversitybasedmergepolicy","intervalbasedmergepolicy","intervalbasedsplitpolicy"], help="the name of the algo")
args = parser.parse_args()

chosen_algo = args.algo_name #the chosen algorithm from args


algos = {
    "stagnationbasedsplitpolicy" : StagnationBasedSplitPolicy(
        patience = 2,
        stagnation_threshold =1e-2,
        split_factor = 3,
        ),
    "randommergepolicy" : RandomMergePolicy(
        patience=5,
        stagnation_threshold = 1e-8
    ),
    "highdiversitybasedmergepolicy" : DiversityBasedMergePolicy(
        patience=5,
        stagnation_threshold = 1e-8,
        merge_on_high_diversity = True,
    ),
    "lowdiversitybasedmergepolicy":  DiversityBasedMergePolicy(
        patience=5,
        stagnation_threshold = 1e-8,
        merge_on_high_diversity = False,
    ),
    "intervalbasedmergepolicy" : IntervalBasedMergePolicy(
        merge_factor=3
    ),
    "intervalbasedsplitpolicy" : IntervalBasedSplitPolicy(
        split_factor=3
    )
}

all_functions = [
    (0, bent_cigar),
    (1, sum_diff_pow),
    (2, zakharov),
    (3, rosenbrock),
    (4, rastrigin),
    (5, expanded_schaffers_f6),
    (6, lunacek_bi_rastrigin),
    (7, non_cont_rastrigin),
    (8, levy),
    (9, modified_schwefel),
    (10, high_conditioned_elliptic),
    (11, discus),
    (12, ackley),
    (13, weierstrass),
    (14, griewank),
    (15, katsuura),
    (16, happy_cat),
    (17, h_g_bat),
    (18, expanded_griewanks_plus_rosenbrock),
    (19, schaffers_f7),
]

DIMS = 100
NUM_RUNS = 10
MAIN_SEED = 11111
#for algo_idx, (algo_name, algo) in enumerate(algos.items()):
algo_name, algo = chosen_algo,algos[chosen_algo]
with open(f"output_{algo_name}.txt","w") as file:
        file.write(f"ALGORITHM: {algo_name}\n")
        for f_id, func in all_functions:
            results = []
            file.write(f"FUNC: f{f_id}\n")
            file.write(f"RUNS:\n[\n")
            for run in range(NUM_RUNS):

                seed = MAIN_SEED + f_id *NUM_RUNS+ run #method to get a unique seed for each run
                np.random.seed(seed)

                if (algo_name == "stagnationbasedsplitpolicy" or algo_name =="intervalbasedsplitpolicy"):
                    pso = dcpso.DCPSO(
                        swarm_size = 100,
                        dims = 100,
                        upper_bounds = [100]*DIMS,
                        lower_bounds = [-100]*DIMS,
                        obj_function = func,
                        num_iterations = 3000,
                        neighborhood_size = 3,
                        split_policy = copy.deepcopy(algo)
                    )
                else:
                     pso = mcpso.MCPSO(
                        swarm_size = 100,
                        dims = 100,
                        upper_bounds = [100]*DIMS,
                        lower_bounds = [-100]*DIMS,
                        obj_function = func,
                        num_iterations = 3000,
                        neighborhood_size = 3,
                        merge_policy = copy.deepcopy(algo)
                     )
                pso.init()
                result, _ = pso.run()
                file.write(f"{result[0]},\n")
                results.append(result)
            file.write("]\n")
            file.write(f"MEAN:{sum(results)/len(results)}\n")


            print(results)

