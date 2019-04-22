#!/usr/bin/env python

import os, sys
import time
import numpy as np  # "pip install numpy" installs numpy
import random
import math
import csv
from numpy.linalg import inv

from envs.ea import de

import gym
from gym import spaces
from gym.utils import seeding

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from callbacks import *
from rl.agents.dqn import DQNAgent
#from rl.policy_copy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.util import *

class PolicyDebug(object):
    """Abstract base class for all implemented policies.
        
        Each policy helps with selection of action to take on an environment.
        
        Do not use this abstract base class directly but instead use one of the concrete policies implemented.
        To implement your own policy, you have to implement the following methods:
        
        - `select_action`
        
        # Arguments
        agent (rl.core.Agent): Agent used
    """
    def _set_agent(self, agent):
        self.agent = agent
    
    @property
    def metrics_names(self):
        return []
    
    @property
    def metrics(self):
        return []
    
    def select_action(self, **kwargs):
        raise NotImplementedError()
    
    def get_config(self):
        """Return configuration of the policy
            
            # Returns
            Configuration as dict
        """
        return {}

class BoltzmannQPolicy(PolicyDebug):
    """Implement the Boltzmann Q Policy
        
        Boltzmann Q Policy builds a probability law on q values and returns
        an action selected randomly according to this law.
    """
    def __init__(self, tau=1., clip=(-500., 500.)):
        super(BoltzmannQPolicy, self).__init__()
        self.tau = tau
        self.clip = clip
    
    def select_action(self, q_values):
        """Return the selected action
            
            # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
            
            # Returns
            Selection action
        """
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]
        
        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values);
        f.write('qvalue = {} '.format(q_values))
        f.write('probability = {} \n'.format(probs)+'\n')
        #print(q_values, probs)
        action = np.random.choice(range(nb_actions), p=probs)
        return action
    
    def get_config(self):
        """Return configurations of BoltzmannQPolicy
            
            # Returns
            Dict of config
        """
        config = super(BoltzmannQPolicy, self).get_config()
        config['tau'] = self.tau
        config['clip'] = self.clip
        return config


class EpsGreedyQPolicy(PolicyDebug):
    """Implement the epsilon greedy policy
        
        Eps Greedy policy either:
        
        - takes a random action with probability epsilon
        - takes current best action with prob (1 - epsilon)
        """
    def __init__(self, eps=.1):
        super(EpsGreedyQPolicy, self).__init__()
        self.eps = eps
    
    def select_action(self, q_values):
        """Return the selected action
            
            # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
            
            # Returns
            Selection action
            """
        assert q_values.ndim == 1
        f.write('qvalue = {} '.format(q_values)+'\n')
        nb_actions = q_values.shape[0]
        
        if np.random.uniform() < self.eps:
            action = np.random.random_integers(0, nb_actions-1)
        else:
            action = np.argmax(q_values)
        return action
    
    def get_config(self):
        """Return configurations of EpsGreedyQPolicy
            
            # Returns
            Dict of config
            """
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config

def DE(fun, lbounds, ubounds, dim, best_value):
    # f = open('qvalue', 'a')
    ENV_NAME = 'ea'

    env = de.DEEnv(fun, lbounds, ubounds, dim, best_value)
    
    nb_actions = env.action_space.n
    #print((1,) + env.observation_space.shape)
    # Build a sequential model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(nb_actions, activation = 'linear'))
    print("Model Summary: ",model.summary())

    #print("keras1...", budget)
    memory = SequentialMemory(limit=100000, window_length=1)#100000
    
    # Boltzmann Q Policy
    #print("Boltzmann Q Policy")
    policy = EpsGreedyQPolicy() #BoltzmannQPolicy()
    
    # DQN Agent: Finally, we configure and compile our agent. You can use every built-in Keras optimizer and even the metrics!
    #print("DQN Agent")
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1e4, target_model_update=1e3, policy=policy, enable_double_dqn = True, batch_size = 64) # nb_steps_warmup >= nb_steps 2000
    # DQN stores the experience i the memory buffer for the first nb_steps_warmup. This is done to get the required size of batch during experience replay.
    # When number of steps exceeds nb_steps_warmup then the neural network would learn and update the weight.

    # Neural Compilation
    #print("Neural Compilation")
    dqn.compile(Adam(lr=1e-4), metrics=['mae'])
    dqn.load_weights('dqn_ea_weights.h5f')

    #log_filename = 'dqn_{}_log.json'.format(ENV_NAME)
    #callbacks = [FileLogger(log_filename, interval=10)]
    #callbacks = [TrainEpisodeLogger()]

    # Fit the model: training for nb_steps = number of generations
    #print("Fit the model ")
    #dqn.fit(env, callbacks = callbacks, nb_steps=115e8, visualize=False, verbose=0, nb_max_episode_steps = None) #int(budget)

    # After training is done, we save the final weights.
    #print("Save the weights ",self.budget)
    #dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    #f.close()
    # Test
    #print("Test")
    dqn.test(env, nb_episodes=25, visualize=False)

