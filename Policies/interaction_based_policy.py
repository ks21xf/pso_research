import numpy as np
from Algorithms.swarm import Swarm
from Utilities.Sobol import Sobol

class InteractionBasedPolicy():
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

    def split_swarms(self, algorithm, interactions):
      """splits the current swarms based on the grouping determined by det_grouping.
      a new swarm object is created from a subset of the current swarm's particle information and global information.
      """

      #leave if number of swarms equals number of dims
      if len(algorithm.swarms) == algorithm.dims:
          return
      print(algorithm.swarms[0])
      print(interactions)
      splits_for_new_swarms = interactions
      new_swarms =[]

      #from the original swarm, split it according to the splits
      for original_swarm_idx, original_swarm in enumerate(splits_for_new_swarms):

          for curr_splits_index, curr_splits in enumerate(original_swarm):
              print(curr_splits)
              indices = np.atleast_1d(algorithm.swarms[original_swarm_idx].indices[curr_splits])
              lower_bounds = np.atleast_1d(algorithm.swarms[original_swarm_idx].lower_bounds[curr_splits])
              upper_bounds = np.atleast_1d(algorithm.swarms[original_swarm_idx].upper_bounds[curr_splits])

              # we know what the dimensions of the new swarm's arrays will be, its (self.swarm_size,indices). so allocate the memory with zeros and fill in the values to avoid creating new arrays a bunch
              new_position = algorithm.swarms[original_swarm_idx].position[:,curr_splits]
              new_velocity = algorithm.swarms[original_swarm_idx].velocity[:,curr_splits]
              new_pbest_pos =algorithm.swarms[original_swarm_idx].pbest_pos[:,curr_splits]
              new_gbest_pos =np.atleast_1d(algorithm.swarms[original_swarm_idx].gbest_pos[curr_splits])

              if new_position.ndim ==1:
                new_position = new_position[:,np.newaxis]
                new_velocity = new_velocity[:,np.newaxis]
                new_pbest_pos = new_pbest_pos[:,np.newaxis]


              #make new swarm
              new_swarms.append(self.create_swarm(algorithm,upper_bounds,lower_bounds,indices))

              new_swarms[-1].position = new_position
              new_swarms[-1].velocity = new_velocity
              new_swarms[-1].pbest_pos = new_pbest_pos
              new_swarms[-1].gbest_pos = np.copy(new_gbest_pos)

      #replace the old swarms with the split swarms and calc fitness
      algorithm.swarms = new_swarms
      #[print(x) for x in new_swarms]
      #input()


    def execute(self,algorithm):
        """executes the specific merge policy"""

        #for first iteration, determine the split intervals based on the algorithm
        if self.iterations == 0:
            self.determine_split_intervals(algorithm)
            print("Determining Interactions...")
            sob = Sobol(dims=algorithm.dims,num_samples=100000,func=algorithm.obj_function)
            interactions_raw, leftovers = sob.second_order()
            interactions = []
            for key,value in interactions_raw.items():
                interactions.append(np.array((key[0],key[1])))
            for ele in leftovers:
                interactions.append(np.array(ele))
            print([interactions])

            self.split_swarms(algorithm,[interactions])
            [print(x) for x in algorithm.swarms]
            print("Interactions found.")

        #split under these conditions
        if(False and self.iterations%self.split_intervals==0 and self.iterations!=0):
            self.split_swarms(algorithm)
            for swarm in algorithm.swarms:
                algorithm.set_fitness_of_swarm(swarm)
        if self.iterations == 2000:
            listy = np.zeros((11,2))
            print(listy)
            for gg in range(11):
                listy[gg] = algorithm.swarms[gg].gbest_pos
            print("igs")
            print(listy)


            for dd in algorithm.swarms:
                print(dd)
        self.iterations+=1