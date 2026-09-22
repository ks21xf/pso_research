import numpy as np
from Algorithms.swarm import Swarm

class DiversityBasedMergePolicy():
  """MCPSO merge policy that merges stagnated swarms with the highest/lowest diversity swarm
    Attributes:
    self.stagnation_threshold - float
        -the threshold of differences in fitness at which a swarm is considered stagnating
    self.patience - int
        -the number of iterations that a swarm is allowed to stagnate until it is eligble for merging
    self.merge_on_high_diversity - boolean
        -determines if stagnated swarms get merged with the swarm with the highest or lowest diversity
    self.stagnation_list - list
        -hold on to swarms that have confirmed stagnated while calculating velocity/position to divide them after"""

  def __init__(self,stagnation_threshold,patience, merge_on_high_diversity,reset_velocity=False):
    self.stagnation_threshold = stagnation_threshold
    self.patience = patience
    self.merge_on_high_diversity = merge_on_high_diversity

    self.stagnation_list = []

    self.reset_velocity = reset_velocity

  def create_swarms(self,algorithm,upper_bounds,lower_bounds,indices):
      return Swarm(
                  swarm_size = algorithm.swarm_size_pso,
                  upper_bounds = upper_bounds,
                  lower_bounds = lower_bounds,
                  indices = indices,
                  neighborhood_size = algorithm.neighborhood_size_pso,
                  stagnation_threshold = self.stagnation_threshold,
      )
  def merge_stagnated_swarms(self,stagnation_list,algorithm):
          """merges the current set of swarms into new swarms according to the grouping."""

          #determine the attributes of the swarm we're about to create
          indices = []
          lower_bounds = []
          upper_bounds = []

          #loop through each swarm's attributes, appending them to a list
          for swarm in stagnation_list:

            indices.extend(swarm.indices)
            lower_bounds.extend(swarm.lower_bounds)
            upper_bounds.extend(swarm.upper_bounds)

          #finally turn them into an np array for performance
          indices = np.array(indices)
          lower_bounds = np.array(lower_bounds)
          upper_bounds = np.array(upper_bounds)

          #make new swarm
          merged_swarm = self.create_swarms(algorithm,upper_bounds,lower_bounds,indices)

          # we know what the dimensions of the new swarm's arrays will be, its (self.swarm_size,indices). so allocate the memory with zeros and fill in the values to avoid creating and allocating memory for new arrays
          new_position = np.zeros((algorithm.swarm_size_pso,indices.shape[0]))
          new_velocity = np.zeros((algorithm.swarm_size_pso,indices.shape[0]))
          new_pbest_pos = np.zeros((algorithm.swarm_size_pso,indices.shape[0]))
          new_gbest_pos = np.zeros(indices.shape[0])
          tracker = 0 #track dims

          #append particle information to the created arrays
          for swarm in stagnation_list:

              new_gbest_pos[tracker:tracker+swarm.dims] = swarm.gbest_pos
              new_position[:,tracker:tracker+swarm.dims] = swarm.position
              new_velocity[:,tracker:tracker+swarm.dims] = 0 if self.reset_velocity else swarm.velocity
              new_pbest_pos[:,tracker:tracker+swarm.dims] = swarm.pbest_pos

              tracker+=swarm.dims

          #assign the new swarms position to the created arrays
          merged_swarm.position = new_position
          merged_swarm.velocity = new_velocity
          merged_swarm.pbest_pos = new_pbest_pos
          merged_swarm.gbest_pos = np.copy(new_gbest_pos)

          #calculate diversity
          algorithm.set_fitness_of_swarm(merged_swarm)
          merged_swarm.calculate_diversity()

          return merged_swarm

  def get_highest_diversity_swarm(self,algorithm):
        best_div = 0
        best_swarm_idx = -1
        for swarm_idx, swarm in enumerate(algorithm.swarms):
            if swarm.diversity > best_div:
                best_div = swarm.diversity
                best_swarm_idx = swarm_idx
        return best_swarm_idx

  def get_lowest_diversity_swarm(self,algorithm):
        lowest_div = np.inf
        lowest_swarm_idx = -1
        for swarm_idx, swarm in enumerate(algorithm.swarms):
            if swarm.diversity < lowest_div:
                lowest_div = swarm.diversity
                lowest_swarm_idx = swarm_idx
        return lowest_swarm_idx

  def calculate_diversity_of_swarms(self,algorithm):
      """calculates diversity of all swarms"""
      for swarm in algorithm.swarms:
        swarm.calculate_diversity()

  def find_stagnated_swarms(self,algorithm):
      """calculates stagnation of all swarms and determines if they should be placed in the stagnation list for merging"""

      #calculate stagnation - if a swarms fitness is considered stagnating for a number of iterations, the swarm is officially stagnated and placed in a separate list.
      next_iteration_of_swarms = []
      for swarm_idx, swarm in enumerate(algorithm.swarms):
          swarm.calculate_stagnation()

          #conditions for a swarm to stagnate. don't allow all swarms to enter the stagnation bucket.
          if swarm.stagnation_counter >= self.patience and len(algorithm.swarms)-1 != len(self.stagnation_list):
              self.stagnation_list.append(swarm)
          else:
              next_iteration_of_swarms.append(swarm)
      algorithm.swarms = next_iteration_of_swarms

  def deal_with_stagnated_swarms(self,algorithm):
      """deals with stagnated swarms based on the policy - determines the lowest diversity swarm and merges the stagnated swarm with that"""

      while self.stagnation_list: #merge whenever there is more than one swarm stagnating

          #choose a swarm and merge a stagnated swarm
          merge_swarm_idx = self.get_highest_diversity_swarm(algorithm) if self.merge_on_high_diversity else self.get_lowest_diversity_swarm(algorithm)
          new_swarm = self.merge_stagnated_swarms([self.stagnation_list[0], algorithm.swarms[merge_swarm_idx]],algorithm)

          #remove the original swarms we just merged
          algorithm.swarms.pop(merge_swarm_idx)
          self.stagnation_list.pop(0)

          #add new swarm
          algorithm.swarms.append(new_swarm)

  def execute(self,algorithm):

    """executes the merge policy"""
    self.calculate_diversity_of_swarms(algorithm)
    self.find_stagnated_swarms(algorithm)
    self.deal_with_stagnated_swarms(algorithm)
