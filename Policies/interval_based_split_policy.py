import numpy as np
from Algorithms.swarm import Swarm

class IntervalBasedSplitPolicy():
    """MCPSO split policy that splits all swarms according to a split factor.
    The split factor determines the intervals at which to split
    Attributes:
    split_factor - int
        -the number by which the swarms divide each splitting phase
    iterations - int
        -policy tracks the number of iterations to determine when splitting occurs
    """
    def __init__(self,split_factor):
        self.split_factor = split_factor
        self.iterations = 0

    def determine_split_intervals(self,algorithm):
        """Determine the intervals of splitting based on the number of iterations and dimensions of the problem space provided by the algorithm"""

        self.split_intervals = np.int64(algorithm.num_iterations/(1+(np.log(algorithm.dims)/np.log(self.split_factor)))) #self.nf

    def create_swarm(self,algorithm, upper_bounds,lower_bounds,indices):
        """creates a swarm object, based on what the split policy needs, some attributes are needed and some arent."""

        return Swarm(
                    swarm_size = algorithm.swarm_size_pso,
                    upper_bounds = upper_bounds,
                    lower_bounds = lower_bounds,
                    indices = indices,
                    neighborhood_size = algorithm.neighborhood_size_pso,
        )

    def det_groupings(self, algorithm):
        """determines the new groupings of the entire swarm. returns a 3d list where:
              second outer most list represents the old swarm about to be broken up.
              inner most lists represent how the indices of the old swarm get split up. e.g [[0,1,2],[3,4,5],[6,7,8]]
              this is wrapped in another list so we have it for multiple swarms."""

        #variables
        indices_list = [] #stores the 3d list of swarm splits
        new_entry = [] #one new entry = new swarm and its splits

        #for every swarm, we want to figure out its splits
        for curr_swarm in algorithm.swarms:
            #we basically want to figure out how to divide the current dims of this swarm as evenly as possible by the split factor.

            #start by finding how many times we can evenly divide dims by split factor
            wholes = curr_swarm.dims//self.split_factor
            remainder = curr_swarm.dims%self.split_factor

            #this covers a case where dims=1
            if wholes == remainder and remainder ==0: continue

            #now we will follow a little algorithm for determining the best splits to make: if d>k, create k splits of d/k each, adding +1 to each split for every d%k. if d<k, there isnt enough dimensions to split by the split factor. so best course of action is to create k swarms.
            dim_tracker = 0
            if curr_swarm.dims >= self.split_factor:
                for j in range(self.split_factor):
                    new_entry.append(curr_swarm.indices[dim_tracker:dim_tracker+wholes+(1 if j<remainder else 0)])
                    dim_tracker+=wholes+1 if j<remainder else wholes
            else: #add each index as its own array
                for i in curr_swarm.indices:
                  new_entry.append(np.array([i]))
            indices_list.append(new_entry)
            new_entry = []

        return indices_list

    def split_swarms(self, algorithm):
      """splits the current swarms based on the grouping determined by det_grouping.
      a new swarm object is created from a subset of the current swarm's particle information and global information.
      """

      #leave if number of swarms equals number of dims
      if len(algorithm.swarms) == algorithm.dims:
          return

      splits_for_new_swarms = self.det_groupings(algorithm)
      new_swarms =[]

      #from the original swarm, split it according to the splits
      for original_swarm_idx, original_swarm in enumerate(splits_for_new_swarms):
          prev_length = 0 #keep track of the lengths of the previous swarm so we can take from the original vector

          for curr_splits_index, curr_splits in enumerate(original_swarm):

              lb,ub = prev_length, prev_length+len(curr_splits) #the bounds of the old swarm where the new swarm will inherit (lower/upper bound). +1 to ub because slicing doesnt include current number
              indices = algorithm.swarms[original_swarm_idx].indices[lb:ub]
              lower_bounds = algorithm.swarms[original_swarm_idx].lower_bounds[lb:ub]
              upper_bounds = algorithm.swarms[original_swarm_idx].upper_bounds[lb:ub]

              # we know what the dimensions of the new swarm's arrays will be, its (self.swarm_size,indices). so allocate the memory with zeros and fill in the values to avoid creating new arrays a bunch
              new_dims = ub-lb
              new_position =  algorithm.swarms[original_swarm_idx].position[:,lb:ub]
              new_velocity = algorithm.swarms[original_swarm_idx].velocity[:,lb:ub]
              new_pbest_pos =algorithm.swarms[original_swarm_idx].pbest_pos[:,lb:ub]
              new_gbest_pos =algorithm.swarms[original_swarm_idx].gbest_pos[lb:ub]


              #make new swarm
              new_swarms.append(self.create_swarm(algorithm,upper_bounds,lower_bounds,indices))

              new_swarms[-1].position = new_position
              new_swarms[-1].velocity = new_velocity
              new_swarms[-1].pbest_pos = new_pbest_pos
              new_swarms[-1].gbest_pos = np.copy(new_gbest_pos)
              prev_length += len(curr_splits)

      #replace the old swarms with the split swarms and calc fitness
      algorithm.swarms = new_swarms

    def execute(self,algorithm):
        """executes the specific merge policy"""

        #for first iteration, determine the split intervals based on the algorithm
        if self.iterations == 0:
            self.determine_split_intervals(algorithm)

        #split under these conditions
        if(self.iterations%self.split_intervals==0 and self.iterations!=0):
            self.split_swarms(algorithm)
            for swarm in algorithm.swarms:
                algorithm.set_fitness_of_swarm(swarm)

        self.iterations+=1