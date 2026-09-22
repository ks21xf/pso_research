import numpy as np
from Policies.interval_based_merge_policy import IntervalBasedMergePolicy
from Policies.diversity_based_merge_policy import DiversityBasedMergePolicy
from Policies.random_merge_policy import RandomMergePolicy

class MCPSO:
    """
    The MCPSO algorithm extends the CPSO algorithm by dynamically changing the number of swarms and the dimensions they work with during execution.
    MCPSO works by dividing all swarms down to its smallest components (like CPSO) and gradually merge swarms by a swarm factor until all swarms merge into a single swarm (PSO)
    Attributes:
    swarm_size - int
      -the number of particles in the swarm
    upper_bounds - ndarray (swarm_size,)
      -the upper bounds of each dimension
    lower_bounds - ndarray (swarm_size,)
      -the lower bounds of each dimension
      dims - int
      -the number of dimensions
    obj_function - function
      -the objective function to optimize
    num_iterations - int
      -the number of iterations to run the algorithm for
    neighborhood_size - int
      -the number of particles in the neighborhood of each particle
    merge_factor - int
      -the number of swarms to merge into a single swarm
    swarms - list of PSOs
      -the list of swarms in the algorithm
    context_vector - ndarray (dims,)
      -the context vector of the algorithm
    nf - int
      -the number of iterations between each merge
    """
    def __init__(self, swarm_size, upper_bounds, lower_bounds, dims, obj_function, num_iterations, neighborhood_size, merge_policy):

        #attributes for CPSO
        self.dims = dims
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds
        self.obj_function = obj_function
        self.num_iterations = num_iterations
        self.swarms = []
        self.merge_policy = merge_policy

        self.context_vector = np.zeros(self.dims)

        #attributes for PSO swarms
        self.swarm_size_pso = swarm_size
        self.neighborhood_size_pso = neighborhood_size

    def create_swarms(self):
        """initialize the desired number of swarms (regular PSO algorithms) based on the dimension.
        basically populate self.swarms and make it a list of PSOs
        this is based on dims because we are creating one swarm per component"""

        #create a swarm for each dimension, including its corresponding bounds and index.
        for i in range(self.dims):

            #create a swarm and append
            self.swarms.append(self.merge_policy.create_swarms(self,[self.upper_bounds[i]],[self.lower_bounds[i]],[i]))

        self._build_context_vector(False)

    def _build_context_vector(self,use_gbest=True):
        """builds context vector with gbests from all swarms. A context vector is a reformed solution of the original space made from getting solutions from all swarms.
        Initially, we have no basis for which to form a context vector, so a random particle is chosen from each swarm to be part of the context vector.
        To test solutions for a swarm, we put its global best position into the its corresponding piece of the context vector and evaluate it while keeping other parts of the context vector constant.
        If this solution results in the context vector preforming better against the objective function, the context vector is updated to include that position.
        This means that the context vector consists of the best solutions from each of the swarms, and current positions for particles are tested by plugging them into their respective spot in the context vector.
        """

        cv = np.zeros(self.dims)

        #build the context vector by adding positions of all swarms together.
        for s in self.swarms:
            if use_gbest:
                cv[s.indices] = s.gbest_pos
            else:
                p = np.random.randint(s.swarm_size)
                cv[s.indices] = s.position[p]
        self.context_vector = np.copy(cv)

    def set_swarm_fitness(self,init=False):
        """sets fitness of the swarms and evaluates each swarms personal and global bests to see if new positions are better.
        If initializing, then we set all particle's pbests to their current position.
        Otherwise, we only update a particle's pbest if their current position is better than their pbest position.
        In addition, if a particle's pbest is better than the swarm's global best, set the swarm's gbest to that particle's pbest."""

        for s in self.swarms:

            #clone the context vector, then reshape to get a context matrix of the same context vector (num_swarm) times.
            context_matrix = np.tile(self.context_vector,self.swarm_size_pso).reshape(self.swarm_size_pso,self.dims) #e have to clone the context vector num_swarm times and put all the particle's positions into the respective spot in the context vector. the reason we'd have to clone the context vector is because keeping one context vector but replacing it each time with the next particle's information would require for loops.

            #replace every relevant part of the context vector with each particle's position
            context_matrix[:,s.indices] = s.position

            #run obj func
            s.fitness = self.obj_function(context_matrix)

            #if we are init, then pbest pos = current pos
            if init:
                s.pbest_pos = s.position.copy()
                s.pbest_fit = s.fitness.copy()
            else: #replace any pbests that are worse than current pos. we want this in an else because there's no point doing this after we init
                improvements = (s.pbest_fit > s.fitness) #find where current fitness is better than the particle's pbest.
                s.pbest_pos = np.where(improvements[:,None],s.position,s.pbest_pos) #numpy being difficult - have to add an extra dimension (d,) to (d,1) to the fitness arrays so this works as intended here since the position is a 2d array whereas fitness is a 1d
                s.pbest_fit = np.where(improvements,s.fitness,s.pbest_fit)

            if s.gbest_fit > np.min(s.pbest_fit):
                new_global_best = np.argmin(s.pbest_fit)
                s.gbest_pos = s.pbest_pos[new_global_best].copy()
                s.gbest_fit = np.min(s.pbest_fit)

                #replace spot in context vector if this is better than current global best
                self.context_vector[s.indices] = s.gbest_pos.copy()

    def set_fitness_of_swarm(self,s):

            #clone the context vector, then reshape to get a context matrix of the same context vector (num_swarm) times.
            context_matrix = np.tile(self.context_vector,self.swarm_size_pso).reshape(self.swarm_size_pso,self.dims) #e have to clone the context vector num_swarm times and put all the particle's positions into the respective spot in the context vector. the reason we'd have to clone the context vector is because keeping one context vector but replacing it each time with the next particle's information would require for loops.

            context_matrix[:,s.indices] = s.pbest_pos
            s.pbest_fit = self.obj_function(context_matrix)

            #replace every relevant part of the context vector with each particle's position
            context_matrix[:,s.indices] = s.position
            s.fitness = self.obj_function(context_matrix)

            #replace any pbests that are worse than current pos. we want this in an else because there's no point doing this after we init
            improvements = (s.pbest_fit > s.fitness) #find where current fitness is better than the particle's pbest.
            s.pbest_pos = np.where(improvements[:,None],s.position,s.pbest_pos) #numpy being difficult - have to add an extra dimension (d,) to (d,1) to the fitness arrays so this works as intended here since the position is a 2d array whereas fitness is a 1d
            s.pbest_fit = np.where(improvements,s.fitness,s.pbest_fit)

            if s.gbest_fit > np.min(s.pbest_fit):
                new_global_best = np.argmin(s.pbest_fit)
                s.gbest_pos = s.pbest_pos[new_global_best].copy()
                s.gbest_fit = np.min(s.pbest_fit)

                #replace spot in context vector if this is better than current global best
                self.context_vector[s.indices] = s.gbest_pos.copy()

    def init(self):
        """initializes algorithm by creating the initial swarms and setting their fitness"""
        self.create_swarms()
        self.set_swarm_fitness(init=True)

    def run(self):
      """runs the algorithm """

      for iteration in range(self.num_iterations):

          #prepare the next iteration of swarms - if swarms are stagnating, remove them from the swarm to merge with other swarms who have also stagnated
          for swarm in self.swarms:

              #calculate the new positions and velocities of all particles from all swarms
              swarm.update_velocity()
              swarm.update_position()

          #determine fitness of all swarms
          self.set_swarm_fitness()

          #run policy
          self.merge_policy.execute(self)

          #print results at the specified interval
          if( iteration == self.num_iterations-1 or iteration % 100 ==0 ): # or iteration % 100 ==0
              print("iteration",iteration)
              print(f"Best Fitness: {self.obj_function(np.atleast_2d(self.context_vector))}")
              print(len(self.swarms))
      return self.obj_function(np.atleast_2d(self.context_vector)), self.context_vector