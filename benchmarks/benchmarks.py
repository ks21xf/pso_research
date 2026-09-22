import numpy as np
def standard_rosenbrock(x: np.ndarray) -> np.ndarray:
    """
    Vectorized N-dimensional Rosenbrock function.

    x.shape = (n_samples, dimensions)
    returns shape = (n_samples,)
    """
    x = np.atleast_2d(x)
    t1 = x[:, 1:] - x[:, :-1] ** 2
    t2 = 1 - x[:, :-1]

    return np.sum(100 * t1**2 + t2**2, axis=1)

def norwegian(x):#-1.1 to 1.1
    first_part = np.cos(np.pi*x**3)
    second_part = (99+x)/100
    return np.multiply((first_part*second_part),axis=1)

def egg_holder(x): #-512 to 512
  first_part = -(x[1]+47)*np.sin(np.sqrt(np.abs(x[0]/(x[1]+47))))
  second_part = -x[0]*np.sin(np.sqrt(np.abs(x[0]/(x[1]+47))))
  return first_part+second_part

def absolute_value(x):
   return np.abs(x,axis=1)