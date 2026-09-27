#goal with this experiment is to demonstrate sobol is a good
#metric for determining variable interaction by testing a
#variety of functions with it
import numpy as np
from Utilities.Sobol import Sobol
from benchmarks.benchmarks import *

#here we will run 20D rosenbrock 100 times, seeing how many times the analysis returns
#all 2-pair variable interactions.
#in addition, we will monitor statistics of the (0,1) interaction, assuming it exists.
#the idea behind that is to see if the sobol analysis is largely consistent across runs

#overarching variables
np.random.seed(11111)
rosenbrock_set = {(0,1), (1,2), (2,3), (3,4), (4,5),(5,6), (6,7), (7,8), (8,9), (9,10), (10,11), (11,12), (12,13), (13,14), (14,15),(15,16), (16,17), (17,18), (18,19)}
empty_set = set([])
correct_runs = 0
num_runs = 100
connection_monitoring = []

#begin main loop
for run in range(num_runs):
    sob = Sobol(dims=20,num_samples=100000,func=standard_rosenbrock)
    connections, leftovers = sob.second_order()

    #add all keys from connections into a set. if it equals the rosenbrock_set, then it
    #found all connections. if leftovers is an empty set, thats further proof.
    current_set = set([])
    for k,v in connections.items():
        current_set.add(k)

    #check if we got the rosenbrock set back and count it as a correct run
    if(current_set==rosenbrock_set) and (leftovers == empty_set): correct_runs+=1

    #save (0,1) connection, if one exists.
    try:
        connection_monitoring.append(connections[(0,1)])
    except KeyError:
        print("no significant (0,1) connection found!")

print("accuracy:",correct_runs/num_runs)
print("(0,1) sobol index mean:",sum(connection_monitoring)/len(connection_monitoring))
print("(0,1) sobol index std:",np.std(np.array(connection_monitoring)))