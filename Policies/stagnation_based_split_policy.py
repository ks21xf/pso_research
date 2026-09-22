import numpy as np
from Algorithms.swarm import Swarm


class StagnationBasedSplitPolicy():
    """DCPSO split policy that splits swarms based on if they are stagnated
    Attributes:
    self.stagnation_threshold - float
        -the threshold of differences in fitness at which a swarm is considered stagnating
    self.patience - int
        -the number of iterations that a swarm is allowed to stagnate until it is eligble for merging
    split_factor - int
        -the number by which the swarms divide each splitting phase
    """
    def __init__(self, patience, stagnation_threshold,split_factor, reset_velocity=False):
        self.patience = patience
        self.stagnation_threshold = stagnation_threshold
        self.split_factor = split_factor

        self.reset_velocity = reset_velocity

    def create_swarm(self,algorithm, upper_bounds,_lower_bounds,indices):
        """initialize the desired number of swarms (regular PSO algorithms) based on the dimension.
        basically populate self.swarms and make it a list of PSOs
        this is based on dims because we are creating one swarm per component"""

        return Swarm(
                    swarm_size = algorithm.swarm_size_pso,
                    upper_bounds = upper_bounds,
                    lower_bounds = _lower_bounds,
                    indices = indices,
                    neighborhood_size = algorithm.neighborhood_size_pso,
                    stagnation_threshold = self.stagnation_threshold
        )

    def split_swarm_of_index(self,i,algorithm):
        """splits the ith swarm in the stagnation list"""

        grouping = self._det_grouping_of_index(i,algorithm)
        prev_length = 0 #keep track of the lengths of the previous swarm so we can take from the original vector
        new_swarms = []

        for curr_splits in grouping:

            lb,ub = prev_length, prev_length+len(curr_splits) #the bounds of the old swarm where the new swarm will inherit (lower/upper bound). +1 to ub because slicing doesnt include current number
            indices = algorithm.swarms[i].indices[lb:ub]
            lower_bounds = algorithm.swarms[i].lower_bounds[lb:ub]
            upper_bounds = algorithm.swarms[i].upper_bounds[lb:ub]

            # we know what the dimensions of the new swarm's arrays will be, its (self.swarm_size,indices). so allocate the memory with zeros and fill in the values to avoid creating new arrays a bunch
            new_position =  algorithm.swarms[i].position[:,lb:ub]
            new_velocity = np.zeros_like(algorithm.swarms[i].velocity[:, lb:ub]) if self.reset_velocity else algorithm.swarms[i].velocity[:,lb:ub]
            new_pbest_pos =algorithm.swarms[i].pbest_pos[:,lb:ub]
            #new_gbest_pos =algorithm.swarms[i].gbest_pos[lb:ub] #dont inherit gbestpos if it gets overidden when fitness evaluating

            #make new swarm
            new_swarms.append(algorithm.split_policy.create_swarm(algorithm,upper_bounds,lower_bounds,indices))

            new_swarms[-1].position = new_position
            new_swarms[-1].velocity = new_velocity
            new_swarms[-1].pbest_pos = new_pbest_pos
            #new_swarms[-1].gbest_pos = np.copy(new_gbest_pos)
            algorithm.set_fitness_of_swarm(new_swarms[-1])
            prev_length += len(curr_splits)

        return new_swarms

    def _det_grouping_of_index(self,i,algorithm):
          """determines the groupings of swarm i in stagnation list"""

          #we basically want to figure out how to divide the current dims of this swarm as evenly as possible by the split factor.
          curr_swarm = algorithm.swarms[i]
          grouping = []

          #start by finding how many times we can evenly divide dims by split factor
          wholes = curr_swarm.dims//algorithm.split_policy.split_factor
          remainder = curr_swarm.dims%algorithm.split_policy.split_factor

          if wholes == remainder and remainder ==0: return []
          #now we will follow a little algorithm for determining the best splits to make: if d>k, create k splits of d/k each, adding +1 to each split for every d%k. if d<k, there isnt enough dimensions to split by the split factor. so best course of action is to create k swarms.
          dim_tracker = 0
          if curr_swarm.dims >= algorithm.split_policy.split_factor:
              for j in range(algorithm.split_policy.split_factor):
                  grouping.append(curr_swarm.indices[dim_tracker:dim_tracker+wholes+(1 if j<remainder else 0)])
                  dim_tracker+=wholes+1 if j<remainder else wholes
          else: #add each index as its own array
              for i in curr_swarm.indices:
                grouping.append(np.array([i]))
          return grouping


    def execute(self, algorithm):
        """executes split policy"""

        new_swarm = []
        for swarm_idx, swarm in enumerate(algorithm.swarms):
            swarm.calculate_stagnation()

            #if swarm stagnates, perform the split and add the new swarms to the next iteration. otherwise, keep the old swarm
            if swarm.stagnation_counter >= self.patience:
                new_swarm.extend(self.split_swarm_of_index(swarm_idx,algorithm))
            else:
                new_swarm.append(swarm)

        algorithm.swarms = new_swarm