
from cec2005real.cec2005 import Function
from numpy.random import rand
import numpy as np
import math
from optproblems import *
from optproblems.cec2005 import *
import validate_dqn

#function no. = 1 to 25
# dim = 2, 10, 30, 50

# Total problem instances = 100

#def EA_AOS(fun, lbounds, ubounds, budget, problem_index):
#print("DE_AOS..",fun)
#cost =
#print("cost: ", cost)
#return cost

# FF = 0.5; CR = 1.0

# d = [2, 10, 30, 50]

# reward_cec1 = np.inf


'''fbench = Function(np.random.randint(1,25), d[np.random.randint(0,3)])
info = fbench.info(); #print(i, info)
dim = info['dimension']
budget = 1e4 * dim
#sol = info['lower']+rand(dim)*(info['upper']-info['lower'])
fun = fbench.get_eval_function()
#print(fun, fun(sol))
lbounds = np.repeat(info['lower'], dim); lbounds = np.array(lbounds)
ubounds = np.repeat(info['upper'], dim); ubounds = np.array(ubounds)
reward_cec = train_dqn.DE(fun, lbounds, ubounds, budget, FF, CR)


while math.fabs(reward_cec - reward_cec1) > 1e-2:
    reward_cec1 = reward_cec'''

d = [10, 30]
func_select = [unimodal.F3, basic_multimodal.F9, f16.F16, f18.F18, f23.F23]

for i in range(2):
    for j in range(5):
        dim = d[i]; print(dim)
        fun = func_select[j](dim)
        lbounds = fun.min_bounds; lbounds = np.array(lbounds); print(lbounds)
        ubounds = fun.max_bounds; ubounds = np.array(ubounds); print(ubounds)
        opti = fun.get_optimal_solutions()
        for o in opti:
            print(o.phenome, o.objective_values)
        sol = np.copy(o.phenome)
        best_value = fun.objective_function(sol)
        #print(" best value= ",best_value)
        #for repeat in range(10):
        validate_dqn.DE(fun, lbounds, ubounds, dim, best_value)
                #best_found += b
        #best_found /= 10
        #print("\n$$$$$$$$$$$$$$$$$$$$$$$$$best value = ",best_value,"mean best found = ", best_found,"$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
        





