import numpy as np
from Algorithms.swarm import Swarm

class RepeatingIntervalBasedPolicy():
    """A universal policy that cycles back from using DCPSO and MCPSO.
    A phase interval determines how many M/DCPSOs we perform using the number of iterations,
        e.g phase interval of 4 with 2000 epochs - we'll do 4 M/DCPSOs, 500 epochs each
    Each M/DPCSO operation uses its alloted epochs to run a standard M/DCPSO.
    When one is done, we switch to the other algorithm, and it picks up where the previous left off.
    """

    def __init__(self,split_factor,merge_factor,num_phases, starting_phase):
        self.split_factor = split_factor
        self.merge_factor = merge_factor
        self.num_phases =num_phases
        self.iterations = 0
        self.phase = starting_phase #'split', 'merge'
        self.operation = self.merge_swarms if starting_phase == 'merge' else self.split_swarms

    def determine_phase_intervals(self,algorithm):
        """Determine the intervals of splitting based on the number of iterations and dimensions of the problem space provided by the algorithm"""
        self.phase_intervals = algorithm.num_iterations//self.num_phases #np.int64(algorithm.num_iterations/(1+(np.log(algorithm.dims)/np.log(self.num_phases))))

    def determine_sub_interval(self,algorithm,factor):
        """determines the merging/splitting intervals for the current phase.
        factor is either merging/splitting factor."""
        self.sub_interval =  np.int64(self.phase_intervals/(1+(np.log(algorithm.dims)/np.log(factor))))

    def create_swarm(self,algorithm, upper_bounds,lower_bounds,indices):
        """creates a swarm object, based on what the split policy needs, some attributes are needed and some arent."""

        return Swarm(
                    swarm_size = algorithm.swarm_size_pso,
                    upper_bounds = upper_bounds,
                    lower_bounds = lower_bounds,
                    indices = indices,
                    neighborhood_size = algorithm.neighborhood_size_pso,
        )
    def det_groupings_for_split(self, algorithm):
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

      splits_for_new_swarms = self.det_groupings_for_split(algorithm)
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

    def det_groupings_for_merge(self,algorithm):
        """determines the new groupings of the entire swarm and returns a list of lists of the old swarms and how they should be merged.
        outer list would contain an inner list of swarms [swarm1,swarm2,swarm3] that indicates those swarms should be merged together to form the new swarm
        the outer list would contain multiples of these inner lists, which would contain all the current swarms and how they should be merged together.
        """

        #variables
        unmerged_swarms = [] #stores a list of lists of swarms. inner list is old swarms that will be merged into a single swarm.
        new_entry = []
        for i in range(len(algorithm.swarms)):

            #if we reached a multiple of the merge factor or are about to leave the loop, that marks the end of a merge
            if (i+1)%self.merge_factor ==0 or i == len(algorithm.swarms)-1:
                new_entry.append(algorithm.swarms[i])
                unmerged_swarms.append(new_entry)
                new_entry = []
            else:
                new_entry.append(algorithm.swarms[i])

        return unmerged_swarms

    def merge_swarms(self,algorithm):
        """merges the current set of swarms into new swarms according to the grouping.
        the det_groupings function above returns an ordering of current swarms. this function's job is to merge them into one swarm.
        a new swarm object is created and the particle information and global information are appended together and carried over into the new swarm."""

        #variables
        unmerged_swarms = self.det_groupings_for_merge(algorithm) #loop through this list of lists, merging the swarms together
        merged_swarms =[] #store the merged swarms here

        #merge all swarms in a group into its own swarm
        for swarm_group in unmerged_swarms:

            #determine the attributes of the swarm we're about to create
            indices = []
            lower_bounds = []
            upper_bounds = []

            #loop through each swarm's attributes, appending them to a list
            for swarm in swarm_group:

                indices.extend(swarm.indices)
                lower_bounds.extend(swarm.lower_bounds)
                upper_bounds.extend(swarm.upper_bounds)

            #finally turn them into an np array for performance
            indices = np.array(indices)
            lower_bounds = np.array(lower_bounds)
            upper_bounds = np.array(upper_bounds)

            #make new swarm
            merged_swarms.append(self.create_swarm(algorithm,upper_bounds,lower_bounds,indices))


            # we know what the dimensions of the new swarm's arrays will be, its (self.swarm_size,indices). so allocate the memory with zeros and fill in the values to avoid creating and allocating memory for new arrays
            new_position = np.zeros((algorithm.swarm_size_pso,indices.shape[0]))
            new_velocity = np.zeros((algorithm.swarm_size_pso,indices.shape[0]))
            new_pbest_pos = np.zeros((algorithm.swarm_size_pso,indices.shape[0]))
            new_gbest_pos = np.zeros(indices.shape[0])

            tracker = 0 #track dims

            #append particle information to the created arrays
            for swarm in swarm_group:
                new_gbest_pos[tracker:tracker+swarm.dims] = swarm.gbest_pos
                new_position[:,tracker:tracker+swarm.dims] = swarm.position
                new_velocity[:,tracker:tracker+swarm.dims] = swarm.velocity
                new_pbest_pos[:,tracker:tracker+swarm.dims] = swarm.pbest_pos
                tracker+=swarm.dims

            #assign the new swarms position to the created arrays
            merged_swarms[-1].position = new_position
            merged_swarms[-1].velocity = new_velocity
            merged_swarms[-1].pbest_pos = new_pbest_pos
            merged_swarms[-1].gbest_pos = np.copy(new_gbest_pos)

        #replace the old swarms with the merged swarms and calc fitness
        algorithm.swarms = merged_swarms

    def execute(self,algorithm):
        """executes the specific merge policy"""

        #for first iteration, determine the split intervals based on the algorithm
        if self.iterations == 0:
            self.determine_phase_intervals(algorithm)
            self.determine_sub_interval(algorithm,self.merge_factor if self.phase == "merge" else self.split_factor)

        #check if it's time for a phase change
        if(self.iterations%self.phase_intervals==0 and self.iterations!=0):
            self.phase = 'merge' if self.phase == "split" else 'split'
            self.operation = self.merge_swarms if self.phase == 'merge' else self.split_swarms
            #determine new merge/split intervals for new m/dcpso
            self.determine_sub_interval(algorithm,self.merge_factor if self.phase == "merge" else self.split_factor)

        #split under these conditions
        if(self.iterations%self.sub_interval==0 and self.iterations!=0):
            #print(self.phase_intervals,self.operation,self.sub_interval)
            self.operation(algorithm)
            for swarm in algorithm.swarms:
                algorithm.set_fitness_of_swarm(swarm)

        self.iterations+=1