import numpy as np
class Swarm:
    """
    The swarm class maintains a group of particle's and their local and current positions along with their velocities and local best fitnesses. Also maintains a global best position and fitness.
    Contains functions for calculating the position and velocity updates of each particle.

    ATTRIBUTES:
    swarm_size - int
        -the number of particles in the swarm
    indices - ndarray (swarm_size,)
        -a unique index of each dimension
    upper_bounds - ndarray (swarm_size,)
        -the upper bounds of each dimension
    lower_bounds - ndarray (swarm_size,)
        -the lower bounds of each dimension
    dims - int
        -the number of dimensions
    neighborhood_size - int
        -the number of particles in the neighborhood of each particle
    gbest_pos - ndarray (dims,)
        -the global best position
    gbest_fit - float
        -the global best fitness
    position - ndarray (swarm_size,dims)
        -the position of each particle
    velocity - ndarray (swarm_size,dims)
        -the velocity of each particle
    fitness - ndarray (swarm_size,)
        -the fitness of each particle
    pbest_pos - ndarray (swarm_size,dims)
        -the personal best position of each particle
    pbest_fit - ndarray (swarm_size,)
        -the personal best fitness of each particle
    c1 - float
        -the cognitive component of the velocity update
    c2 - float
        -the social component of the velocity update
    w - float
        -the inertia weight of the velocity update
    stagnation_threshold - float
        -the number in which the swarm will be considered stagnating if changes in fitness don't exceed
    prev_gbest_fit - float
        -the previous iteration's global best fitness
    stagnation_counter - int
        -the swarm tracks how long fitness changes are below the stagnation threshold
    diversity - float
        -a value that represents the diversity of the swarm. diversity = 0 means no diversity
    pdt - float
        - stands for particle deletion threshold - the minimum distance a particle can have with another particle before it gets deleted
    """
    def __init__(self, swarm_size, indices, upper_bounds, lower_bounds, neighborhood_size,stagnation_threshold=np.inf,pdt=-1):

        #attributes for swarm
        self.swarm_size = swarm_size
        self.indices = np.array(indices)
        self.dims = len(indices)
        self.upper_bounds = np.array(upper_bounds)
        self.lower_bounds = np.array(lower_bounds)
        self.neighborhood_size = neighborhood_size

        self.gbest_pos = np.zeros(self.dims)
        self.gbest_fit = np.inf

        #define particle attributes
        self.position = np.random.uniform(low=self.lower_bounds, high=self.upper_bounds, size=(self.swarm_size,self.dims)) #create random positions by generating numbers inbetween the bounds for each particle (row). a cell in a row is one of the dimensions. the nth column is each particle's nth dimension

        self.velocity = np.zeros((self.swarm_size,self.dims))
        self.fitness = np.full(fill_value=np.inf, shape=self.swarm_size)
        self.pbest_pos = np.zeros((self.swarm_size,self.dims))
        self.pbest_fit = np.full(fill_value=np.inf, shape=self.swarm_size)

        #velocity calculation attributes
        self.c1 = 1.49445 #cognitive component
        self.c2 = 1.49445 #social component
        self.w = 0.729 #inertia weight

        #stagnation tracking
        self.prev_gbest_fit = None
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_counter = 0

        #diversity
        self.diversity = -1 #-1 means unassigned
        self.distance_matrix = None
        self.pdt = pdt

    def __str__(self):
        return f"""PSO Swarm:
        Indices: {self.indices},
        Dimensions: {self.dims},
        Lower Bounds: {self.lower_bounds},
        Upper Bounds: {self.upper_bounds},
        Neighborhood Size: {self.neighborhood_size},
        Num Particles: {self.swarm_size},
        Global Best: {self.gbest_pos},
        Global Best Fitness: {self.gbest_fit},
        Diversity: {self.diversity}
        """

    def print_particle(self,i):
        """
        mainly for debugging, a quick way to see a particle
        i - index of particle to print
        """

        print(f"""Particle {i}:
        Position: {self.position[i]},
        Velocity: {self.velocity[i]},
        Fitness: {self.fitness[i]},
        Personal Best Pos: {self.pbest_pos[i]},
        Personal Best Fit: {self.pbest_fit[i]},
        """)

    def update_velocity(self):
        """calculate the velocity of all particles. Uses the following equation for calculating velocity:
        v = w*v + (c1*r1) * (pbest_pos-position) + (c2*r2) * (lbest_pos-position),
        where:

        v = the particle's velocity
        lbest = the particle's local best position (aka the best position in its group)
        pbest_pos = the particle's personal best position
        position = the particle's current position
        r1, r2 = random numbers between 0 and 1
        w = the inertia weight added to prevent overshooting
        c1, c2 = the cognitive and social components of the velocity update
        """

        #establish some variables to simplify formula
        r1 = np.random.uniform(size =(self.swarm_size,self.dims))
        r2 = np.random.uniform(size = (self.swarm_size,self.dims))
        yhat = self.get_local_bests()
        y = self.pbest_pos
        x = self.position

        self.velocity = self.w*self.velocity + (self.c1*r1) * (y-x) + (self.c2*r2) * (yhat-x)

    def update_position(self):
        """calculate position by simply adding velocity, along with constricting it to the bounded area."""
        self.position += self.velocity
        self.position = np.clip(a=self.position, a_min=self.lower_bounds, a_max=self.upper_bounds) #clip to bounds

    def get_local_bests(self):
        """get the best position of a particle's (represented by the index) group. using window slider method, a particle is in a group with itself and n-1 indices above it (with wrapping)
        where n is the number of group members.

        EXAMPLE OF WINDOW SLIDER METHOD, swarm_size = 5, neighorhood_size = 3. '[ ]' denotes particles, '*' denotes particles within particle i's group. this is how we would determine the local best of particle i.
        for i =0:
        [*][*][*][ ][ ]
        for i =1:
        [ ][*][*][*][ ]
        for i=2:
        [ ][ ][*][*][*]
        for i=3:
        [*][ ][ ][*][*]
        for i=4:
        [*][*][ ][ ][*]
        We would start at particle i and go neighborhood_size-1 steps forward, with wrapping included. (neighborhood_size-1 since we include particle i in the group as well.)
        So to extract the personal bests of each particle, we must get its index and ns-1 indices ahead of it, with wrapping. That is what the code below does.
        """

        #get the indices of every particle's group members
        indices = np.arange(start = 0, stop = self.swarm_size) #particle's current index
        group_indices = (indices[:, None] + np.arange(self.neighborhood_size)) % self.swarm_size

        #determine each particle's local best - the local best will be the index with the smallest personal best fitness
        relative_local_bests = np.argmin(self.pbest_fit[group_indices],axis=1) #because of how argmin works, this gets us the lowest index relative to group_indices.
        actual_local_bests = group_indices[indices, relative_local_bests] #this is the indices of each swarm's local bests
        return self.pbest_pos[actual_local_bests] #get the positions of each of the actual local bests

    def calculate_stagnation(self):
        """determines if the swarm is stagnating by checking if the differences in current and previous iteration's fitnesses
        is below the threshold for stagnation."""

        #stagnation isnt tracked for the first iteration of the swarm, or in other words, when there is no previous gbest_fit
        if self.prev_gbest_fit is None:
            self.prev_gbest_fit = self.gbest_fit
            return

        #if we detect stagnation, increase stagnation counter. but once stagnation stops, reset patience
        if np.abs(self.gbest_fit - self.prev_gbest_fit) < self.stagnation_threshold:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        #replace prev best fit
        self.prev_gbest_fit = np.copy(self.gbest_fit)



    def calculate_diversity(self):
        """Calculates the diversity of the swarm by finding the average pairwise Euclidean distances of the swarms.
        The distances have to be scaled because avg distance increases along with dimensions, so any high/low diversity algorithms
        prioritize maximizing/minimizing dimensions respectively without scaling.
        Distances typically increase by sqrt(d) for each dimension.
        We should divide by sqrt(d) at the end to account for this.
        Also use minmax scaling to get coords between [0,1] -> prevent distances from exploding as dims get very large"""

        #perform minmax scaling
        scaled_pos =(self.position - self.lower_bounds) / (self.upper_bounds - self.lower_bounds) #this broadcasting should work?

        #creates a euclidean distance matrix of particle positions
        row_sums = np.sum(scaled_pos**2, axis=1,keepdims=True)
        dist = row_sums +row_sums.T - 2*scaled_pos @ scaled_pos.T #euclidean distance trick
        dist = np.sqrt(np.clip(dist,a_min = 0,a_max=None)) #because catastrophic cancellation can produce very small negative values, clip values between 0 and infinity

        #take the average of the distance matrix. first do column-wise summations and ignore the distance to self by subtracting one from the denominator.
        self.diversity = (np.mean(np.sum(dist,axis=1)/(len(dist)-1)))/np.sqrt(self.dims) if len(dist) != 1 else 0

    def determine_particle_distance(self):
        """determines the pairwise distances of all particles using a Euclidean Distance Matrix"""

        row_sums = np.sum(self.position**2, axis=1,keepdims=True)
        self.distance_matrix = row_sums +row_sums.T - 2*self.position @ self.position.T #euclidean distance trick
        self.distance_matrix = np.sqrt(np.clip(self.distance_matrix,a_min = 0,a_max=None)) #because catastrophic cancellation can produce very small negative values, clip values between 0 and infinity

    def delete_particles(self):
        """determines if any particles will be getting deleted. checks if any distances are below the pdt and deletes those particles"""

        #get a tuple of swarm indices where the distance matrix is less than the threshold
        #this includes self overlaps and duplicates since EDMs are symmetric
        all_overlapping_swarms = np.where(self.distance_matrix < 1e-2)
        all_overlapping_swarms
        #the output is a tuple where t[0][n] and t[1][n] are overlapping indices.
        #if a particle overlaps with multiple other particles, it will have duplicate entries.

        #remove self overlapping indices
        indices_without_self_overlap = np.where(all_overlapping_swarms[0]!=all_overlapping_swarms[1])

        #get both sides of the overlapping particles and only take swarms that dont self overlap
        overlapping_particles1 = all_overlapping_swarms[0][indices_without_self_overlap]
        overlapping_particles2 = all_overlapping_swarms[1][indices_without_self_overlap]

        #goal is to remove any particles that overlap with another.
        #create an (n,2) array such that any row is [overlapping_particles[n],overlapping_particles2[n]], to facilitate the process
        overlapping_particles_matrix = np.stack((overlapping_particles1,overlapping_particles2)).T

        #get the maxes of each row of the overlapping particle matrix
        #this tells us which particles of each row in the overlapping particle matrix to remove.
        #since out of the overlapping particles, we keep the one with the best fitness. the argmax corresponds to the worst particle in a row.
        indices_of_particles_to_delete = np.argmax(self.fitness[overlapping_particles_matrix],axis=1)

        #this will give the indices of the particles to remove.
        #remove duplicates with np.unique
        chopping_block = np.unique(overlapping_particles_matrix[np.arange(0,len(overlapping_particles_matrix)),indices_of_particles_to_delete])
        return chopping_block