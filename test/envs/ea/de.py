from __future__ import division
from cec2005real.cec2005 import Function
import numpy as np
from numpy.random import rand
import gym
from gym import spaces
from gym.utils import seeding
import math
from scipy.spatial import distance
import time
from scipy.stats import rankdata
from collections import Counter
from optproblems import *
from optproblems.cec2005 import *

def rand1(population, samples, scale, best, i): # DE/rand/1
    r0, r1, r2 = samples[:3]
    return (population[r0] + scale * (population[r1] - population[r2]))

def rand2(population, samples, scale, best, i): # DE/rand/2
    r0, r1, r2, r3, r4 = samples[:5]
    return (population[r0] + scale * (population[r1] - population[r2] + population[r3] - population[r4]))

def rand_to_best2(population, samples, scale, best, i): # DE/rand-to-best/2
    r0, r1, r2, r3, r4 = samples[:5]
    return (population[r0] + scale * (population[best] - population[r0] + population[r1] - population[r2] + population[r3] - population[r4]))

def current_to_rand1(population, samples, scale, best, i): # DE/current-to-rand/1
    r0, r1, r2 = samples[:3]
    return (population[i] + scale * (population[r0] - population[i] + population[r1] - population[r2]))

def select_samples(popsize, candidate, number_samples):
    """
    obtain random integers from range(popsize),
    without replacement.  You can't have the original candidate either.
    """
    idxs = list(range(popsize))
    idxs.remove(candidate)
    return(np.random.choice(idxs, 5, replace = False))

'''
def OM_Update(self):
    third_dim = []
    # success = np.zeros(self.n_ops); unsuccess = np.zeros(self.n_ops)
    for i in range(self.popsize):
        second_dim = np.zeros(7);
        if self.F1[i] <= self.F[i]:
            # success[self.opu[i]] += 1
            second_dim[0] = self.opu[i]
            second_dim[1] = np.exp(-self.F1[i])
            second_dim[2] = self.F[i] - self.F1[i]
            if self.F1[i] <= np.min(self.F):
                second_dim[3] = np.min(self.F) - self.F1[i]
            else:
                second_dim[3] = -1
            if self.F1[i] <= self.best_so_far:
                second_dim[4] = self.best_so_far - self.F1[i]
            else:
                second_dim[4] = -1
            if self.F1[i] <= np.median(self.F):
                second_dim[5] = np.median(self.F) - self.F1[i]
            else:
                second_dim[5] = -1
            second_dim[6] = (self.best_so_far / self.F1[i]) * math.fabs(self.F1[i] - self.F[i])
                
            if np.any(self.window[:, 1] == np.inf):
                for value in range(self.window_size-1,-1,-1):
                    if self.window[value][0] == -1:
                        self.window[value] = second_dim
                        #print(self.window)
                        break
            else:
                for nn in range(self.window_size-1,-1,-1):
                    if self.window[nn][0] == self.opu[i]:
                        for nn1 in range(nn, 0, -1):
                            self.window[nn1] = self.window[nn1-1]
                        self.window[0] = second_dim
                        break
                    elif nn==0 and window[nn][0] != opu[i]:
                        if F1[i] < np.max(window[: ,1]):
                            window[np.argmax(window[:,1])] = second_dim
            X[i][:] = u[i][:]
            F[i] = self.F1[i]
            third_dim.append(second_dim);# print("r, rule: ",r, rule)
        else:
            # unsuccess[self.opu[i]] += 1
            second_dim = [-1 for i in range(7)]
            second_dim[0] = opu[i]
            third_dim.append(second_dim)

    #print("Outside ", self.window)
    gen_window.append(third_dim); # print("gen_window= ",self.gen_window, type(self.gen_window), np.shape(self.gen_window))
    return gen_window, window
'''


def min_max(a, mi, mx):
    if a < mi:
        mi = a
    if a > mx:
        mx = a
    return mi, mx

def normalise(a, mi, mx):
    a = (a - mi) / (mx - mi);
    return a

def count_success(popsize, gen_window, i, j, Off_met):
    c_s = 0; c_us = 0
    #for k in range(popsize):
        #if gen_window[j, k, 0] == i and gen_window[j, k, Off_met] != -1:
            #c_s += 1
        #if gen_window[j, k, 0] == i and gen_window[j, k, Off_met] == -1:
            #c_us += 1
    c_s = np.sum((gen_window[j, :, 0] == i) & (gen_window[j, :, Off_met] != -1))
    c_us = np.sum((gen_window[j, :, 0] == i) & (gen_window[j, :, Off_met] == -1))
    return c_s, c_us

def count_op(n_ops, window, Off_met): # ???????Include ranking for minimising case??????? Use W-r in place r
    # Gives rank to window[:, Off_met]: largest number will get largest number rank
    rank = rankdata(window[:, Off_met].round(1), method = 'min')
    order = rank.argsort()
    # order gives the index of rank in ascending order. Sort operators and rank in increasing rank.
    window_op_sorted = window[order, 0];
    rank = rank[order]
    rank = rank[window_op_sorted >= 0]
    window_op_sorted = window_op_sorted[window_op_sorted >= 0]; # print("window_op_sorted = ",window, window_op_sorted, rank, order)
    
    # counts number of times an operator is present in the window
    N = np.zeros(n_ops); # print(N, window_op_sorted)
    # the number of times each operator appears in the sliding window
    op, count = np.unique(window_op_sorted, return_counts=True); # print(op, count)
    for i in range(len(count)):
        # print(len(count), i, op[i], count[i])
        N[int(op[i])] = count[i]
    return window_op_sorted, N, rank
                                                        ##########################Success based###########################################

# Applicable for fix number of generations
# Parameter(s): max_gen
def Success_Rate1(popsize, n_ops, gen_window, Off_met, max_gen):
    state_value = np.zeros(n_ops)
    gen_window = np.array(gen_window)
    if len(gen_window) < max_gen:
        max_gen = len(gen_window)
    for i in range(n_ops):
        appl = 0; t_s = 0
        for j in range(len(gen_window)-1, len(gen_window)-max_gen-1, -1):
            total_success = 0; total_unsuccess = 0
            #print("first: ",gen_window, np.shape(gen_window), len(gen_window), i, j);print(gen_window[j,0,0])
            if np.any(gen_window[j, :, 0] == i):
                total_success, total_unsuccess = count_success(popsize, gen_window, i, j, Off_met)
                t_s += total_success
                appl += total_success + total_unsuccess
        if appl != 0:
            state_value[i] = t_s / appl
        #else:
            #state_value[i] = 0
    return state_value

                                                        ##########################Weighted offspring based################################

# Applicable for fix number of generations
# Parameter(s): max_gen
def Weighted_Offspring1(popsize, n_ops, gen_window, Off_met, max_gen):
    state_value = np.zeros(n_ops)
    gen_window = np.array(gen_window)
    if len(gen_window) < max_gen:
        max_gen = len(gen_window)
    for i in range(n_ops):
        appl = 0
        for j in range(len(gen_window)-1, len(gen_window)-max_gen-1, -1):
            total_success = 0; total_unsuccess = 0
            #print("first: ",gen_window, np.shape(gen_window), len(gen_window), i, j);print(gen_window[j,0,0])
            if np.any(gen_window[j, :, 0] == i):
                total_success, total_unsuccess = count_success(popsize, gen_window, i, j, Off_met)
                state_value[i] += np.sum(gen_window[j, np.where((gen_window[j, :, 0] == i) & (gen_window[j, :, Off_met] != -1)), Off_met])
                appl += total_success + total_unsuccess
        if appl != 0:
            state_value[i] = state_value[i] / appl
    if np.sum(state_value) != 0:
        state_value = state_value / np.sum(state_value)
    return state_value

# index: 8 Applicable for fix window size
# Parameter(s): window
def Weighted_Offspring2(popsize, n_ops, window, Off_met, max_gen):
    state_value = np.zeros(n_ops)
    window = window[window[:, 0] != -1][:, :]; # print("weighted offspring2", window)
    # window_op_sorted, N, rank = count_op(n_ops, window, Off_met)
    for i in range(n_ops):
        # print(i, np.sum((window[:, 0] == i) & (window[:, Off_met] != -1)))
        if np.sum((window[:, 0] == i) & (window[:, Off_met] != -1)) > 0:
            # print(np.sum(window[np.where((window[:, 0] == i) & (window[:, Off_met] != -1)), Off_met]), np.sum((window[:, 0] == i) & (window[:, Off_met] != -1)))
            state_value[i] = np.sum(window[np.where((window[:, 0] == i) & (window[:, Off_met] != -1)), Off_met]) / np.sum((window[:, 0] == i) & (window[:, Off_met] != -1)); # print(i, state_value[i])
    # print("state value for off met", Off_met, "is: ", state_value)
    if np.sum(state_value) != 0:
        state_value = state_value / np.sum(state_value)
    # print("normalised state value ", state_value)
    return state_value

                                                        ##########################Best offspring based#############################

# Applicable for fix number of generations
# Parameter(s): max_gen
def Best_Offspring1(popsize, n_ops, gen_window, Off_met, max_gen):
    state_value = np.zeros(n_ops)
    gen_window = np.array(gen_window)
    best_t = np.zeros(n_ops); best_t_1 = np.zeros(n_ops)
    for i in range(n_ops):
        # for last 2 generations
        n_applications = np.zeros(2)
        # Calculating best in current generation
        if np.any((gen_window[len(gen_window)-1, :, 0] == i) & (gen_window[len(gen_window)-1, :, Off_met] != -1)):
            total_success, total_unsuccess = count_success(popsize, gen_window, i, len(gen_window)-1, Off_met)
            n_applications[0] = total_success + total_unsuccess
            # if np.any(gen_window[len(gen_window)-1, :, Off_met] != -1):
            # print(i, Off_met, gen_window, gen_window[len(gen_window)-1, :, Off_met], np.where((gen_window[len(gen_window)-1, :, 0] == i) & (gen_window[len(gen_window)-1, :, Off_met] != -1)))
            best_t[i] = np.max(gen_window[len(gen_window)-1, np.where((gen_window[len(gen_window)-1, :, 0] == i) & (gen_window[len(gen_window)-1, :, Off_met] != -1)), Off_met]); # print(i, best_t[i])
        # Calculating best in last generation
        if len(gen_window)>=2 and np.any((gen_window[len(gen_window)-2,:,0] == i) & (gen_window[len(gen_window)-2, :, Off_met] != -1)):
            total_success, total_unsuccess = count_success(popsize, gen_window, i, len(gen_window)-2, Off_met)
            n_applications[1] = total_success + total_unsuccess
            # if np.any(gen_window[len(gen_window)-2, :, Off_met] != -1):
            best_t_1[i] = np.max(gen_window[len(gen_window)-2, np.where((gen_window[len(gen_window)-2, :, 0] == i) & (gen_window[len(gen_window)-2, :, Off_met] != -1)), Off_met]); # print(i, best_t_1[i])
        if best_t_1[i] != 0 and np.fabs(n_applications[0] - n_applications[1]) != 0:
            state_value[i] = np.fabs(best_t[i] - best_t_1[i]) / ((best_t_1[i]) * (np.fabs(n_applications[0] - n_applications[1])))
        elif best_t_1[i] != 0 and np.fabs(n_applications[0] - n_applications[1]) == 0:
            state_value[i] = np.fabs(best_t[i] - best_t_1[i]) / (best_t_1[i])
        elif best_t_1[i] == 0 and np.fabs(n_applications[0] - n_applications[1]) != 0:
            state_value[i] = np.fabs(best_t[i] - best_t_1[i]) / (np.fabs(n_applications[0] - n_applications[1]))
        else:
            state_value[i] = np.fabs(best_t[i] - best_t_1[i])
    if np.sum(state_value) != 0:
        state_value = state_value / np.sum(state_value)
    return state_value

# Applicable for fix number of generations
# Parameter(s): max_gen
def Best_Offspring2(popsize, n_ops, gen_window, Off_met, max_gen):
    state_value = np.zeros(n_ops)
    gen_window = np.array(gen_window)
    if len(gen_window) < max_gen:
        max_gen = len(gen_window)
    for i in range(n_ops):
        gen_best = []
        for j in range(len(gen_window)-1, len(gen_window)-max_gen-1, -1):
            # print("first: ", i, j, np.hstack(gen_window[j, np.where((gen_window[j,:,0] == i) & (gen_window[j, :, Off_met] != -1)), Off_met]))
            if np.any((gen_window[j,:,0] == i) & (gen_window[j, :, Off_met] != -1)):
                gen_best.append(np.max(np.hstack(gen_window[j, np.where((gen_window[j,:,0] == i) & (gen_window[j, :, Off_met] != -1)), Off_met])))
                state_value[i] += np.sum(gen_best); # print(state_value[i])
        # if gen_best != []:
            # state_value[i] = (1/max_gen) * state_value[i] / np.max(gen_best)
    if np.sum(state_value) != 0:
        state_value = state_value / np.sum(state_value)
    return state_value


                                                ##########################class DEEnv###########################################

mutations = [rand1, rand2, rand_to_best2, current_to_rand1]

class DEEnv(gym.Env):
    def __init__(self, fun, lbounds, ubounds, dim, best_value):
        self.n_ops = 4
        self.action_space = spaces.Discrete(self.n_ops)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(99,), dtype = np.float32)
        self.max_gen = 10
        self.window_size = 50
        self.number_metric = 5
        self.FF = 0.5; self.CR = 1.0
        self.fun = fun
        self.lbounds = lbounds; lbounds = np.array(lbounds); #print(self.lbounds)
        self.ubounds = ubounds; ubounds = np.array(ubounds)
        self.dim = dim
        self.best_value = best_value
        self.best_found = 0; self.c = 0;
        #self.func_select = [unimodal.F1, unimodal.F2, unimodal.F3, unimodal.F4, unimodal.F5, basic_multimodal.F6, basic_multimodal.F8, basic_multimodal.F9, basic_multimodal.F10, basic_multimodal.F11, basic_multimodal.F12, expanded_multimodal.F13, expanded_multimodal.F14, f15.F15, f16.F16, f17.F17, f18.F18, f19.F19, f20.F20, f21.F21, f22.F22, f23.F23, f24.F24]
        #self.func_select = [unimodal.F1, unimodal.F2, unimodal.F5, basic_multimodal.F6, basic_multimodal.F8, basic_multimodal.F10, basic_multimodal.F11, basic_multimodal.F12, expanded_multimodal.F13, expanded_multimodal.F14, f15.F15, f19.F19, f20.F20, f21.F21, f22.F22, f24.F24]
        #self.d = [2, 10, 30, 50]
        #self.optima = [-450.0, -450.0, -310.0, 390.0, -140.0, -330.0,  90.0, 3.141592653589793, -130.0, -300.0, 120.0, 10.0, 10.0, 360.0, 360.0, 260.0]
        #self.file = open('statistic_output', 'a')
    
    def step(self, action):
        #assert self.action_space.contains(action); #print(action)
        self.opu[self.i] = action
        mutate = mutations[action]
    
        # Evolution of parent i
        bprime = mutate(self.population, self.r, self.FF, self.best, self.i)
        bprime[np.where(bprime < self.lbounds[0])] = self.lbounds[0]
        bprime[np.where(bprime > self.ubounds[0])] = self.ubounds[0]
        #if np.any((bprime < self.lbounds[0]) | (bprime > self.ubounds[0])):
            #print("out of bounds",bprime)
        
        self.crossovers = (np.random.rand(self.dim) < self.CR)
        self.crossovers[self.fill_points[self.i]] = True
        self.u[self.i][:] = np.where(self.crossovers, bprime, self.X[self.i])
        
        self.F1[self.i] = self.fun(self.u[self.i])
        # print(self.u[self.i], self.F1[self.i], self.best_so_far, self.best_value, self.fun)
        # a= time.time()
        reward = 0
        second_dim = np.zeros(self.number_metric);
        if self.F1[self.i] <= self.copy_F[self.i]:
            second_dim[0] = self.opu[self.i]
            # second_dim[1] = self.F1[self.i]
            second_dim[1] = self.copy_F[self.i] - self.F1[self.i]
            if self.F1[self.i] < np.min(self.copy_F):
                second_dim[2] = np.min(self.copy_F) - self.F1[self.i]
            else:
                second_dim[2] = -1
            if self.F1[self.i] < self.best_so_far:
                second_dim[3] = self.best_so_far - self.F1[self.i]
                self.best_so_far = self.F1[self.i]
                self.best_so_far_position = self.population[self.i]
                self.stagnation_count = 0;
                reward = 10
            else:
                second_dim[3] = -1
                reward = 1
                self.stagnation_count += 1
            if self.F1[self.i] < np.median(self.copy_F):
                second_dim[4] = np.median(self.copy_F) - self.F1[self.i]
            else:
                second_dim[4] = -1
            # FIFO window
            if np.any(self.window[:, 1] == np.inf):
                for value in range(self.window_size-1,-1,-1):
                    if self.window[value][0] == -1:
                        self.window[value] = second_dim
                        #print(self.window)
                        break
            else:
                for nn in range(self.window_size-1,-1,-1):
                    if self.window[nn][0] == self.opu[self.i]:
                        for nn1 in range(nn, 0, -1):
                            self.window[nn1] = self.window[nn1-1]
                            self.window[0] = second_dim
                            break
                    elif nn==0 and self.window[nn][0] != self.opu[self.i]:
                        if (self.copy_F[self.i] - self.F1[self.i]) < np.max(self.window[: ,1]):
                            self.window[np.argmax(self.window[:,1])] = second_dim
            #print("Inside step..",self.window)
            if self.worst_so_far < self.F1[self.i]:
                self.worst_so_far = self.F1[self.i]
            self.F[self.i] = self.F1[self.i]
            self.X[self.i] = self.u[self.i]
            self.third_dim.append(second_dim)
        else:
            second_dim = [-1 for i in range(self.number_metric)]
            second_dim[0] = self.opu[self.i]
            self.third_dim.append(second_dim)
        
        self.max_std = np.std((np.repeat(self.best_so_far, self.NP/2), np.repeat(self.worst_so_far, self.NP/2)));#print("best so far after: ",self.F1[self.i], self.F[self.i], self.best_so_far)
    
        self.budget -= 1
        self.i = self.i+1

        if self.i >= self.NP:
            #self.file.write('{} generation finished \n'.format(self.generation));
            self.gen_window.append(self.third_dim);
            self.copy_ob = np.zeros(64)
            # Generation based statistics
            self.copy_ob[0:4] = Success_Rate1(self.NP, self.n_ops, self.gen_window, 1, self.max_gen); # print(self.ob[19:23])
            self.copy_ob[4:8] = Success_Rate1(self.NP, self.n_ops, self.gen_window, 2, self.max_gen)
            self.copy_ob[8:12] = Success_Rate1(self.NP, self.n_ops, self.gen_window, 3, self.max_gen)
            self.copy_ob[12:16] = Success_Rate1(self.NP, self.n_ops, self.gen_window, 4, self.max_gen)
            
            self.copy_ob[16:20] = Weighted_Offspring1(self.NP, self.n_ops, self.gen_window, 1, self.max_gen)
            self.copy_ob[20:24] = Weighted_Offspring1(self.NP, self.n_ops, self.gen_window, 2, self.max_gen)
            self.copy_ob[24:28] = Weighted_Offspring1(self.NP, self.n_ops, self.gen_window, 3, self.max_gen)
            self.copy_ob[28:32] = Weighted_Offspring1(self.NP, self.n_ops, self.gen_window, 4, self.max_gen)
            
            self.copy_ob[32:36] = Best_Offspring1(self.NP, self.n_ops, self.gen_window, 1, self.max_gen)
            self.copy_ob[36:40] = Best_Offspring1(self.NP, self.n_ops, self.gen_window, 2, self.max_gen)
            self.copy_ob[40:44] = Best_Offspring1(self.NP, self.n_ops, self.gen_window, 3, self.max_gen)
            self.copy_ob[44:48] = Best_Offspring1(self.NP, self.n_ops, self.gen_window, 4, self.max_gen)
            
            self.copy_ob[48:52] = Best_Offspring2(self.NP, self.n_ops, self.gen_window, 1, self.max_gen)
            self.copy_ob[52:56] = Best_Offspring2(self.NP, self.n_ops, self.gen_window, 2, self.max_gen)
            self.copy_ob[56:60] = Best_Offspring2(self.NP, self.n_ops, self.gen_window, 3, self.max_gen)
            self.copy_ob[60:64] = Best_Offspring2(self.NP, self.n_ops, self.gen_window, 4, self.max_gen)
            
            self.third_dim = []
            self.opu = np.zeros(self.NP) * 4
            self.i = 0;
            self.fill_points = np.random.randint(self.dim, size = self.NP); #print("fill points", self.fill_points)
            self.generation = self.generation + 1
            self.population = np.copy(self.X)
            self.copy_F = np.copy(self.F)
            self.best = np.argmin(self.copy_F)
            self.pop_average = np.average(self.copy_F);
            self.pop_std = np.std(self.copy_F)
        
        # Preparation for observation to give for next action decision
        self.r = select_samples(self.NP, self.i, 5); #print("value of r1: ",r)
        self.jrand = np.random.randint(self.dim)

        ob = np.zeros(99); ob[19:83] = np.copy(self.copy_ob)
        # a = time.time()
        # Parent fintness
        ob[0] = normalise(self.copy_F[self.i], self.best_so_far, self.worst_so_far); #print(self.ob[0])
        # Population fitness statistic
        ob[1] = normalise(self.pop_average, self.best_so_far, self.worst_so_far); #print(self.ob[1])
        ob[2] = self.pop_std / self.max_std; #print(self.ob[2])
        ob[3] = self.budget / self.max_budget; #print(self.ob[3])
        ob[4] = self.dim / 50; # print(self.ob[4])
        ob[5] = self.stagnation_count / self.max_budget; # print(self.best_so_far, self.ob[5])
        # Random sample based observations
        ob[6:12] = distance.cdist(self.population[[self.r[0],self.r[1],self.r[2],self.r[3],self.r[4],self.best]], np.expand_dims(self.population[self.i], axis=0)).T / self.max_dist
        ob[12:18] = np.fabs(self.copy_F[[self.r[0],self.r[1],self.r[2],self.r[3],self.r[4],self.best]] - self.copy_F[self.i]) / (self.worst_so_far - self.best_so_far)
        ob[18] = distance.euclidean(self.best_so_far_position, self.population[self.i]) / self.max_dist;
        
        # Window based statistics
        ob[83:87] = Weighted_Offspring2(self.NP, self.n_ops, self.window, 1, self.max_gen)
        ob[87:91] = Weighted_Offspring2(self.NP, self.n_ops, self.window, 2, self.max_gen)
        ob[91:95] = Weighted_Offspring2(self.NP, self.n_ops, self.window, 3, self.max_gen)
        ob[95:99] = Weighted_Offspring2(self.NP, self.n_ops, self.window, 4, self.max_gen)
        
        # print("state: ",time.time() - a)
        
        #self.file.write(','.join([str(i) for i in self.ob])+'\n')
        if self.budget <= 0:
            #self.file.write('Episode finished \n');
            print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$",self.budget, self.best_value, self.best_so_far,"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            self.c += 1
            self.best_found += self.best_so_far
            if self.c == 25:
                print("\n$$$$$$$$$$$$$$$$$$$$$$$$$best value = ",self.best_value,"mean best found = ", self.best_found / self.c ,"$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            # file.close()
            return ob, reward, True, {}
        else:
            return ob, reward, False, {}


    def reset(self):
        #fbench = Function(np.random.randint(6,26), self.d[np.random.randint(1, 3)])
        #self.info = fbench.info(); print(self.info)
        #self.dim = self.info['dimension']
        #self.fun = fbench.get_eval_function()
        #self.lbounds = np.repeat(self.info['lower'], self.dim); self.lbounds = np.array(self.lbounds)
        #self.ubounds = np.repeat(self.info['upper'], self.dim); self.ubounds = np.array(self.ubounds)
        #self.best_value = self.info['best']
        
        #self.dim = self.d[np.random.randint(1, 3)]; print(self.dim)
        #self.rand_fun = np.random.randint(0,16); #self.rand_fun = 7
        #self.fun = self.func_select[self.rand_fun](self.dim)
        #self.lbounds = self.fun.min_bounds; self.lbounds = np.array(self.lbounds); print(self.lbounds)
        #self.ubounds = self.fun.max_bounds; self.ubounds = np.array(self.ubounds); print(self.ubounds)
        #self.best_value = self.optima[self.rand_fun]; print(self.best_value)
        #opti = self.fun.get_optimal_solutions()
        #for o in opti:
            #print(o.phenome, o.objective_values)
            #sol = np.copy(o.phenome)
        #self.best_value = self.fun.objective_function(sol)

        #print("Function info: fun= ", self.rand_fun, " best value= ", self.best_value)
        self.budget = 1e4 #* self.dim
        self.max_budget = 1e4 #* self.dim
        self.generation = 0
        self.NP = 100 #* self.dim
        self.X = self.lbounds + ((self.ubounds - self.lbounds) * np.random.rand(self.NP, self.dim))
        self.F = [self.fun(x) for x in self.X]
        self.u = [[0 for z in range(int(self.dim))] for k in range(int(self.NP))]
        self.F1 = np.zeros(int(self.NP));
        self.budget -= self.NP
        # Make changes to X wherever needed using u and use popultion to pick random solutions
        self.population = np.copy(self.X)
        self.copy_F = np.copy(self.F)
    
        self.population = np.copy(self.X)
        self.best_so_far = np.min(self.copy_F);
        self.best_so_far_position = self.population[np.argmin(self.copy_F)]
        self.worst_so_far = np.max(self.copy_F);
        #if np.any(self.copy_F < self.best_value):
            #print(self.X, self.copy_F, self.best_so_far, self.info)
        self.i = 0;
        self.r = select_samples(self.NP, self.i, 5)
        self.best = np.argmin(self.F)
        self.jrand = np.random.randint(self.dim)
        
        self.window = [[np.inf for j in range(self.number_metric)] for i in range(self.window_size)]; self.window = np.array(self.window); self.window[:, 0].fill(-1)
        self.gen_window = []
        self.third_dim = []
        self.opu = np.zeros(self.NP) * 4
        
        # Randomly selects from [0,dim-1] of size NP
        self.fill_points = np.random.randint(self.dim, size = self.NP)
        
        self.pop_average = np.average(self.copy_F)
        self.pop_std = np.std(self.copy_F)
        
        #if self.best_so_far < self.best_value:
            #print(self.best_so_far_position, self.best_so_far, self.best_value)
        
        ob = np.zeros(99); self.copy_ob = np.zeros(64)
        
        self.max_dist = distance.euclidean(self.lbounds, self.ubounds)
        self.max_std = np.std((np.repeat(self.best_so_far, self.NP/2), np.repeat(self.worst_so_far, self.NP/2))); #print(self.max_std)
        self.stagnation_count = 0;
        return ob


