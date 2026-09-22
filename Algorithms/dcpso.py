import numpy as np
from Policies.interval_based_split_policy import IntervalBasedSplitPolicy
from Policies.stagnation_based_split_policy import StagnationBasedSplitPolicy
import pandas as pd

class DCPSO:
    """
    The DCPSO algorithm extends the CPSO algorithm by dynamically changing the number of swarms and the dimensions they work with during execution.
    DCPSO works similarly except it starts with all swarms merged already (PSO) and divides the swarm into smaller and smaller swarms until eventually dividing all swarms down to its smallest components (like CPSO)
    Attributes:
    swarm_size - int
      -the number of particles in the swarm
    upper_bounds - ndarray (dims,)
      -the upper bounds of each dimension
    lower_bounds - ndarray (dims,)
      -the lower bounds of each dimension
      dims - int
      -the number of dimensions
    obj_function - function
      -the objective function to optimize
    num_iterations - int
      -the number of iterations to run the algorithm for
    neighborhood_size - int
      -the number of particles in the neighborhood of each particle
    split_factor - int
      -the number of times to divide a swarm
    swarms - list of PSOs
      -the list of swarms in the algorithm
    context_vector - ndarray (dims,)
      -the context vector of the algorithm
    nf - int
      -the number of iterations between each merge
    """
    def __init__(self, swarm_size, upper_bounds, lower_bounds, dims, obj_function, num_iterations, neighborhood_size, split_policy):

        #attributes for CPSO
        self.dims = dims
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds
        self.obj_function = obj_function
        self.num_iterations = num_iterations
        self.swarms = []
        self.split_policy = split_policy

        self.context_vector = np.zeros(self.dims)

        #attributes for PSO swarms
        self.swarm_size_pso = swarm_size
        self.neighborhood_size_pso = neighborhood_size

    def create_swarm(self):
        """initializes the algorithm as a PSO - since no swarm splitting is happening yet, we just initialize as a regular pso"""


        self.swarms.append(self.split_policy.create_swarm(self,self.upper_bounds,self.lower_bounds,list(range(self.dims))))
        self._build_context_vector(False)

    def _build_context_vector(self,use_gbest=True):
        """builds context vector with gbests from all swarms. A context vector is a reformed solution of the original space made from getting solutions from all swarms.
        Initially, we have no basis for which to form a context vector, so a random particle is chosen from each swarm to be part of the context vector.
        To test solutions for a swarm, we put its global best position into the its corresponding piece of the context vector and evaluate it while keeping other parts of the context vector constant.
        If this solution results in the context vector preforming better against the objective function, the context vector is updated to include that position.
        This means that the context vector consists of the best solutions from each of the swarms, and current positions for particles are tested by plugging them into their respective spot in the context vector.
        """
        cv = np.zeros(self.dims)
        for s in self.swarms:
            #build the context vector by adding positions of all swarms together.
            if use_gbest:
                cv[s.indices] = s.gbest_pos
            else:
                p = np.random.randint(s.swarm_size)
                cv[s.indices] = s.position[p]
        self.prev_bests = np.copy(self.context_vector)
        self.context_vector = np.copy(cv)

    def init(self):
        """initializes algorithm by creating the initial swarms and setting their fitness"""
        self.create_swarm()
        for s in self.swarms:
          self.set_fitness_of_swarm(s,init=True)


    def set_fitness_of_swarm(self,s,init=False):

            #clone the context vector, then reshape to get a context matrix of the same context vector (num_swarm) times.
            context_matrix = np.tile(self.context_vector,self.swarm_size_pso).reshape(self.swarm_size_pso,self.dims) #e have to clone the context vector num_swarm times and put all the particle's positions into the respective spot in the context vector. the reason we'd have to clone the context vector is because keeping one context vector but replacing it each time with the next particle's information would require for loops.

            context_matrix[:,s.indices] = s.pbest_pos
            s.pbest_fit = self.obj_function(context_matrix)

            #replace every relevant part of the context vector with each particle's position
            context_matrix[:,s.indices] = s.position
            s.fitness = self.obj_function(context_matrix)

            #replace any pbests that are worse than current pos. we want this in an else because there's no point doing this after we init
            if init:
                s.pbest_pos = s.position.copy()
                s.pbest_fit = s.fitness.copy()
            else:
                improvements = (s.pbest_fit > s.fitness) #find where current fitness is better than the particle's pbest.
                s.pbest_pos = np.where(improvements[:,None],s.position,s.pbest_pos) #numpy being difficult - have to add an extra dimension (d,) to (d,1) to the fitness arrays so this works as intended here since the position is a 2d array whereas fitness is a 1d
                s.pbest_fit = np.where(improvements,s.fitness,s.pbest_fit)

            if s.gbest_fit > np.min(s.pbest_fit):
                new_global_best = np.argmin(s.pbest_fit)
                s.gbest_pos = s.pbest_pos[new_global_best].copy()
                s.gbest_fit = np.min(s.pbest_fit)

                #replace spot in context vector if this is better than current global best
                self.context_vector[s.indices] = s.gbest_pos.copy()

    def run(self):
      """runs the algorithm """

      for iteration in range(self.num_iterations):

          #calculate the new positions and velocities of all particles from all swarms
          for swarm_idx, swarm in enumerate(self.swarms):
              swarm.update_velocity()
              swarm.update_position()

          #determines fitness of all swarms
          for s in self.swarms:
              self.set_fitness_of_swarm(s)

          #run the splitting policy
          self.split_policy.execute(self)

          #print results at this arbitrary interval
          if(iteration%100==0 or iteration == self.num_iterations-1):
              print("iteration",iteration)
              print(f"Best Fitness: {self.obj_function(np.atleast_2d(self.context_vector))}")
              print(len(self.swarms))

      return self.obj_function(np.atleast_2d(self.context_vector)), self.context_vector
