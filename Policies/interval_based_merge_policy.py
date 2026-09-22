import numpy as np
from Algorithms.swarm import Swarm

class IntervalBasedMergePolicy():
  """MCPSO merge policy that merges all swarms according to a merge factor.
  The merge factor determines the intervals at which to merge
  Attributes:
  merge_factor - int
    -the number of swarms that get merged together during a merging phase"""
  def __init__(self,merge_factor):
      self.merge_factor = merge_factor
      self.iterations = 0


  def determine_merge_intervals(self,algorithm):
      #determine intervals of merging
      self.merge_intervals = np.int64(algorithm.num_iterations/(1+(np.log(algorithm.dims)/np.log(self.merge_factor)))) #self.nf

  def create_swarms(self,algorithm, upper_bounds,_lower_bounds,indices):
        """initialize the desired number of swarms (regular PSO algorithms) based on the dimension.
        basically populate self.swarms and make it a list of PSOs
        this is based on dims because we are creating one swarm per component"""

        return Swarm(
                    swarm_size = algorithm.swarm_size_pso,
                    upper_bounds = upper_bounds,
                    lower_bounds = _lower_bounds,
                    indices = indices,
                    neighborhood_size = algorithm.neighborhood_size_pso,
        )
  def det_groupings(self,algorithm):
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
      unmerged_swarms = self.det_groupings(algorithm) #loop through this list of lists, merging the swarms together
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
          merged_swarms.append(self.create_swarms(algorithm,upper_bounds,lower_bounds,indices))


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

      if self.iterations == 0:
          self.determine_merge_intervals(algorithm)

      #merge under these conditions
      if(self.iterations%self.merge_intervals==0 and self.iterations!=0 and len(algorithm.swarms)>1):
          self.merge_swarms(algorithm)

      self.iterations+=1