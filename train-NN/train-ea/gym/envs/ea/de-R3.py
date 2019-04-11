from __future__ import division
import numpy as np
import random
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
    c_s = np.sum((gen_window[j, :, 0] == i) & (gen_window[j, :, Off_met] != -1))
    c_us = np.sum((gen_window[j, :, 0] == i) & (gen_window[j, :, Off_met] == -1))
    return c_s, c_us

                                                        ##########################Success based###########################################

# Applicable for fix number of generations
def Success_Rate1(popsize, n_ops, gen_window, Off_met, max_gen):
    state_value = np.zeros(n_ops)
    gen_window = np.array(gen_window)
    if len(gen_window) < max_gen:
        max_gen = len(gen_window)
    for i in range(n_ops):
        appl = 0; t_s = 0
        for j in range(len(gen_window)-1, len(gen_window)-max_gen-1, -1):
            total_success = 0; total_unsuccess = 0
            if np.any(gen_window[j, :, 0] == i):
                total_success, total_unsuccess = count_success(popsize, gen_window, i, j, Off_met)
                t_s += total_success
                appl += total_success + total_unsuccess
        if appl != 0:
            state_value[i] = t_s / appl
    return state_value

                                                        ##########################Weighted offspring based################################

# Applicable for fix number of generations
def Weighted_Offspring1(popsize, n_ops, gen_window, Off_met, max_gen):
    state_value = np.zeros(n_ops)
    gen_window = np.array(gen_window)
    if len(gen_window) < max_gen:
        max_gen = len(gen_window)
    for i in range(n_ops):
        appl = 0
        for j in range(len(gen_window)-1, len(gen_window)-max_gen-1, -1):
            total_success = 0; total_unsuccess = 0
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
def Weighted_Offspring2(popsize, n_ops, window, Off_met, max_gen):
    state_value = np.zeros(n_ops)
    window = window[window[:, 0] != -1][:, :]
    for i in range(n_ops):
        if np.sum((window[:, 0] == i) & (window[:, Off_met] != -1)) > 0:
            state_value[i] = np.sum(window[np.where((window[:, 0] == i) & (window[:, Off_met] != -1)), Off_met]) / np.sum((window[:, 0] == i) & (window[:, Off_met] != -1))
    if np.sum(state_value) != 0:
        state_value = state_value / np.sum(state_value)
    return state_value

                                                        ##########################Best offspring based#############################

# Applicable for fix number of generations
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
            best_t[i] = np.max(gen_window[len(gen_window)-1, np.where((gen_window[len(gen_window)-1, :, 0] == i) & (gen_window[len(gen_window)-1, :, Off_met] != -1)), Off_met])
        # Calculating best in last generation
        if len(gen_window)>=2 and np.any((gen_window[len(gen_window)-2,:,0] == i) & (gen_window[len(gen_window)-2, :, Off_met] != -1)):
            total_success, total_unsuccess = count_success(popsize, gen_window, i, len(gen_window)-2, Off_met)
            n_applications[1] = total_success + total_unsuccess
            best_t_1[i] = np.max(gen_window[len(gen_window)-2, np.where((gen_window[len(gen_window)-2, :, 0] == i) & (gen_window[len(gen_window)-2, :, Off_met] != -1)), Off_met])
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
def Best_Offspring2(popsize, n_ops, gen_window, Off_met, max_gen):
    state_value = np.zeros(n_ops)
    gen_window = np.array(gen_window)
    if len(gen_window) < max_gen:
        max_gen = len(gen_window)
    for i in range(n_ops):
        gen_best = []
        for j in range(len(gen_window)-1, len(gen_window)-max_gen-1, -1):
            if np.any((gen_window[j,:,0] == i) & (gen_window[j, :, Off_met] != -1)):
                gen_best.append(np.max(np.hstack(gen_window[j, np.where((gen_window[j,:,0] == i) & (gen_window[j, :, Off_met] != -1)), Off_met])))
                state_value[i] += np.sum(gen_best)
    if np.sum(state_value) != 0:
        state_value = state_value / np.sum(state_value)
    return state_value


                                                ##########################class DEEnv###########################################

mutations = [rand1, rand2, rand_to_best2, current_to_rand1]

class DEEnv(gym.Env):
    def __init__(self):
        self.n_ops = 4
        self.action_space = spaces.Discrete(self.n_ops)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(99,), dtype = np.float32)
        self.max_gen = 10
        self.window_size = 50
        self.number_metric = 5
        self.FF = 0.5; self.CR = 1.0
        self.func_choice = [unimodal.F1, unimodal.F2, unimodal.F5, basic_multimodal.F6, basic_multimodal.F8, basic_multimodal.F10, basic_multimodal.F11, basic_multimodal.F12, expanded_multimodal.F13, expanded_multimodal.F14, f15.F15, f19.F19, f20.F20, f21.F21, f22.F22, f24.F24]
        self.d_choice = [10, 30]
        self.func_select = [(func, d) for func in self.func_choice for d in self.d_choice]
        self.fun_index = 0
    
    def step(self, action):
        self.opu[self.i] = action
        mutate = mutations[action]
    
        # Evolution of parent i
        bprime = mutate(self.population, self.r, self.FF, self.best, self.i)
        bprime[np.where(bprime < self.lbounds[0])] = self.lbounds[0]
        bprime[np.where(bprime > self.ubounds[0])] = self.ubounds[0]
        
        self.crossovers = (np.random.rand(self.dim) < self.CR)
        self.crossovers[self.fill_points[self.i]] = True
        self.u[self.i][:] = np.where(self.crossovers, bprime, self.X[self.i])
        
        self.F1[self.i] = self.fun(self.u[self.i])
        reward = 0
        second_dim = np.zeros(self.number_metric);
        if self.F1[self.i] <= self.copy_F[self.i]:
            second_dim[0] = self.opu[self.i]
            second_dim[1] = self.copy_F[self.i] - self.F1[self.i]
            reward = (self.copy_F[self.i] - self.F1[self.i]) / (self.F1[self.i] - self.best_value + 0.001)
            if self.F1[self.i] < np.min(self.copy_F):
                second_dim[2] = np.min(self.copy_F) - self.F1[self.i]
            else:
                second_dim[2] = -1
            if self.F1[self.i] < self.best_so_far:
                second_dim[3] = self.best_so_far - self.F1[self.i]
                self.best_so_far = self.F1[self.i]
                self.best_so_far_position = self.population[self.i]
                self.stagnation_count = 0;
            else:
                second_dim[3] = -1
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
            if self.worst_so_far < self.F1[self.i]:
                self.worst_so_far = self.F1[self.i]
            self.F[self.i] = self.F1[self.i]
            self.X[self.i] = self.u[self.i]
            self.third_dim.append(second_dim)
        else:
            second_dim = [-1 for i in range(self.number_metric)]
            second_dim[0] = self.opu[self.i]
            self.third_dim.append(second_dim)
        
        self.max_std = np.std((np.repeat(self.best_so_far, self.NP/2), np.repeat(self.worst_so_far, self.NP/2)))
        
        self.budget -= 1
        self.i = self.i+1
        
        if self.i >= self.NP:
            self.gen_window.append(self.third_dim);
            # Generation based statistics
            self.copy_ob = np.zeros(64)
            self.copy_ob[0:4] = Success_Rate1(self.NP, self.n_ops, self.gen_window, 1, self.max_gen)
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
            self.fill_points = np.random.randint(self.dim, size = self.NP)
            self.generation = self.generation + 1
            self.population = np.copy(self.X)
            self.copy_F = np.copy(self.F)
            self.best = np.argmin(self.copy_F)
            self.pop_average = np.average(self.copy_F);
            self.pop_std = np.std(self.copy_F)
        
        # Preparation for observation to give for next action decision
        self.r = select_samples(self.NP, self.i, 5)
        self.jrand = np.random.randint(self.dim)

        ob = np.zeros(99); ob[19:83] = np.copy(self.copy_ob)
        # Parent fintness
        ob[0] = normalise(self.copy_F[self.i], self.best_so_far, self.worst_so_far)
        # Population fitness statistic
        ob[1] = normalise(self.pop_average, self.best_so_far, self.worst_so_far)
        ob[2] = self.pop_std / self.max_std
        ob[3] = self.budget / self.max_budget
        ob[4] = self.dim / 50
        ob[5] = self.stagnation_count / self.max_budget
        # Random sample based observations
        ob[6:12] = distance.cdist(self.population[[self.r[0],self.r[1],self.r[2],self.r[3],self.r[4],self.best]], np.expand_dims(self.population[self.i], axis=0)).T / self.max_dist
        ob[12:18] = np.fabs(self.copy_F[[self.r[0],self.r[1],self.r[2],self.r[3],self.r[4],self.best]] - self.copy_F[self.i]) / (self.worst_so_far - self.best_so_far)
        # Best candidate seen so far
        ob[18] = distance.euclidean(self.best_so_far_position, self.population[self.i]) / self.max_dist
        
        # Window based statistics
        ob[83:87] = Weighted_Offspring2(self.NP, self.n_ops, self.window, 1, self.max_gen)
        ob[87:91] = Weighted_Offspring2(self.NP, self.n_ops, self.window, 2, self.max_gen)
        ob[91:95] = Weighted_Offspring2(self.NP, self.n_ops, self.window, 3, self.max_gen)
        ob[95:99] = Weighted_Offspring2(self.NP, self.n_ops, self.window, 4, self.max_gen)

        if self.budget <= 0:
            return ob, reward, True, {}
        else:
            return ob, reward, False, {}


    def reset(self):
        self.dim = self.func_select[self.fun_index][1]
        self.fun = self.func_select[self.fun_index][0](self.dim)
        self.lbounds = self.fun.min_bounds; self.lbounds = np.array(self.lbounds); print(self.lbounds)
        self.ubounds = self.fun.max_bounds; self.ubounds = np.array(self.ubounds); print(self.ubounds)
        opti = self.fun.get_optimal_solutions()
        for o in opti:
            sol = np.copy(o.phenome)
        self.best_value = self.fun.objective_function(sol)

        print("Function info: fun= ", self.func_select[self.fun_index], " with dim = ", self.dim, " with best value= ", self.best_value)
        self.budget = 1e4
        self.max_budget = 1e4
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
        
        ob = np.zeros(99); self.copy_ob = np.zeros(64)
        
        self.max_dist = distance.euclidean(self.lbounds, self.ubounds)
        self.max_std = np.std((np.repeat(self.best_so_far, self.NP/2), np.repeat(self.worst_so_far, self.NP/2)))
        self.stagnation_count = 0
        
        self.fun_index += 1
        if self.fun_index == len(self.func_select):
            self.fun_index = 0
            random.shuffle(self.func_select)
        
        return ob


