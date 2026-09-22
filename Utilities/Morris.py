#morris good copy
import numpy as np
class Morris:
  def __init__(self,p,trajectories,dims,lower_bounds,upper_bounds,func,relative_mu_threshold):
    """Implementation of the Morris Method:

    The morris method is a senstivity analysis method that determines which input
    variables influence a function's output.

    Note that while it determines which variables are highly influencial, this
    method does not deduce which input variables influence others, it merely
    determines which variables highly influence the OUTPUT.

    As such, the Morris method is used as a filtering mechanism to weed out
    variables without influence.

    How the method works:
    Start by treating the input space like a grid. p specifies how the grid is made up.
    e.g a p = 4 means the grid is [0, 0.333, 0.667, 1].
    The goal behind this is to get a good representation of the input space.
    if the function is not within [0,1] bounds, translate the grid point by reversing minmax scaling: p*(ub-lb)+lb
    Let's say the dims of the function we wish to use this method on is d.
    we will randomly generate d grid points, one for each decision variable
      so for p=4, we could have [0.33,0.67,0.33,...,0]. lets call it x0
    now our goal is to see when we make changes to one of the points, how the output changes
    we run the current arrangement of points into the function and get f0
    then we perturb x1
      that means we just go to the next gridpoint. (we can + or - 0.33 to current gridpoint to do this )
      (but because we want more dramatic changes, we actually double it and use 0.66)
      so we perturb x1, it becomes .33+.67 =1
      we stick [1,0.67,0.33,...,0] (x1) through the function and get f1
    we use the formula f0-f1/x0-x1 to determine what we call the Elementary Effects (EE) of x1
    EE represents x1's effects on the final output (can be positive/negative, represents output going up/down)
    and keep going. get f2 and x2 by perturbing x2 (built off x1) to get EE of x2
    once you get to xi, you completed what is called one trajectory. you have to run enough
    trajectories to get a good idea of the true elementary effects.
    after sufficient trajectories (20-30), you will have 20-30 collections of EE for all xi.
    you can average them, but you can have positive/negative EE which can skew the mean.
    typically absolute mean (mu*) is taken, along with regular std.
    typically a high mu* or high std can indicate a variable that influences the output more.

    ATTRIBUTES:
    p - value of which the grid is divided
    trajectories - the number of tests preformed on each decision variable
    dims - dimensions of function
    func - function in question
    delta - the magnitude of the perturbation
    relative_mu_threshold - decision variable that are at least this percentage of the largest mu* are considered influential"""

    #set vars
    self.p = p
    self.trajectories = trajectories
    self.dims = dims
    self.lower_bounds = lower_bounds
    self.upper_bounds = upper_bounds
    self.func = func
    self.delta = p/(2*(p-1))
    self.relative_mu_threshold = relative_mu_threshold

    #results
    self.mu = -1
    self.mu_star = -1
    self.std =-1

    #determine grid based on p - linspace determines p intervals between 0 and 1
    self.grid = np.linspace(0,1,p)

    #we will have (dims,trajectories) array to story EEs for all xs
    self.EE = np.zeros((trajectories,dims))

  def scale_to_bounds(self,array):
      """scales the array of gridpoints to the specified upper and lower bounds
      (basically undoing minmax scaling)"""
      return array*(self.upper_bounds-self.lower_bounds)+self.lower_bounds

  def run(self,):
    """runs the simulation, returning filtered indices of what it seems the most significant variables"""
    for i in range(self.trajectories):

        #randomly generate the original set of points on the grid
        original = np.random.choice(np.linspace(0,1,self.p),(self.dims))
        original_op = self.func(original)

        #determine pertubations to be made. if larger than 0.5, we're going backwards, else we go forwards, because we're tryna stay within bounds
        pertubations = np.where(original >0.5,-self.delta,self.delta)

        #generate inputs for the current trajectory. create a 2d matrix of the original, one row for each dim. the next row will have the next perturbation.
        current_traj = np.tile(original,(self.dims,1))[:,np.arange(0,len(original))]

        #apply the perturbations to all dims.because its one at a time (OAT), we use loop.
        for d in range(self.dims):
            indices = np.arange(len(current_traj),0+d,-1)-1
            current_traj[indices,d] = current_traj[indices,d]+pertubations[d]

        #stick the original output with the output of the trajectories
        results = np.concatenate((self.func(self.scale_to_bounds(original)),self.func(self.scale_to_bounds(current_traj))))

        #we get EEs for this trajectory by taking f(i) - f(i-1)
        deltas = original-current_traj[-1]
        EE_for_traj = (results[1:] - results[:-1])/(deltas*(self.upper_bounds-self.lower_bounds))

        #storing them row-wise in a 2d matrix means we can get xi's EE by taking EE[:,i]
        self.EE[i] = EE_for_traj

    self.mu = np.mean(self.EE,axis=0) #mean
    self.mu_star = np.mean(np.abs(self.EE),axis=0) #absolute mean (mu star)
    self.sigma = np.sqrt(np.sum(((self.EE - self.mu).T)**2,axis=1)*(1/(self.trajectories-1))) #std

    return self.filter()

  def filter(self):
    """uses relative filtering to remove variables considered noninfluential
    Morris method is qualitative - traditional way to do this is graph mu* and sigma
    and determine what is influential and what is not based on intuition.
    there are some other ways to do this more objectively:
    - sigma=mu* as the benchmark, anything below that line is considered noninfluential
    - relative/percentile filtering, which involves filtering out variables whose
    mu* and sigma is below, lets say 90%, of the max mu*."""

    relative_filter_value = np.max(self.mu_star) *(1-self.relative_mu_threshold)
    return np.where(self.mu_star >= relative_filter_value)[0]


# morris= Morris(
#     p = 4,
#     trajectories = 60,
#     dims = 5,
#     lower_bounds = np.array([-30]*5),
#     upper_bounds = np.array([30]*5),
#     func = func,
#     relative_mu_threshold = 0.9
# )
# print(morris.run())
# print(morris.mu_star)