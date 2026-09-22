import numpy as np
class Sobol:
  """The Sobol class takes in a function, the number of dimensions and the number of samples wanted (higher = more accuracy)
  and determines the variable interactions"""
  def __init__(self,dims,num_samples,func):
    self.dims = dims
    self.num_samples = num_samples
    self.func = func

    #setup
    rng = np.random.default_rng()
    self.A =  rng.uniform(0,3,(num_samples,dims))
    self.B = rng.uniform(0,3,(num_samples,dims))
    self.Y_A = func(self.A)
    self.Y_B = func(self.B)

    #create Y_AB which is Y_AB^(i) such that i is exluded
    self.Y_AB =[]
    for i in range(self.dims):
      AB = self.A.copy()
      AB[:,i] = self.B[:,i]
      self.Y_AB.append(func(AB))
    self.Y_AB =np.array(self.Y_AB)

    #calc stats
    combined_output = np.concat((self.Y_A,self.Y_B))
    self.mean = np.mean(combined_output)
    self.var = np.var(combined_output)

  def total_order(self):
    ST_x = []
    for i in range(self.dims):
      ST_x.append( (1/self.num_samples)*(np.sum((self.Y_A-self.Y_AB[i])**2))/(2*self.var))
    return ST_x

  def first_order(self):
    S_x=[]
    for i in range(self.dims):
      S_x.append(((1/self.num_samples)*np.sum(self.Y_B*self.Y_AB[i]-self.mean**2))/var)
    return S_x

  def second_order(self):
    """calculates second order variable interactions for the function, returning them as a dict with
    key being the indices in the interaction (i,j) with the value being the associated strength of
    interaction: (i,j) -> 0.1. only significant interactions are allowed to proceed.

    To ensure all variables get through, a set of indices is maintained. indices get removed from this
    set when they are found in a significant interaction. indices remaining in the set after all
    interactions are inspected get added by themselves.
      -(basically, if some variables have no interactions, they won't make it past the filtering.
      so this part ensures noninfluential variables actually make it in.)"""

    #create a set of all variable indices
    idx_set = set([])
    idx_set.update(list(range(0,self.dims)))

    #second order setup
    Y_xy = {}
    for i in range(self.dims-1):
      for j in range(i+1,self.dims):
          AB = self.A.copy()
          AB[:,[i,j]] = self.B[:,[i,j]]
          Y_xy[(i,j)] = self.func(AB)

    #calculations and storage
    S_xy= {}
    for key,value in Y_xy.items():
         ans = np.abs((
            np.mean(self.Y_B*value) -
            np.mean(self.Y_B*self.Y_AB[key[0]]) -
            np.mean(self.Y_B*self.Y_AB[key[1]]) +
            np.mean(self.Y_B*self.Y_A)
        )/self.var)

         #save the answer if it is not zero
         if ans > 10e-12:
          S_xy[key] = ans

          idx_set-= {key[0],key[1]} #remove influential variables from the set

    #return the interactions along with the leftovers in the set
    return S_xy, idx_set