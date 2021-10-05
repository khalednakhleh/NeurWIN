'''
Environment to calculate the Whittle index values as a deep reinforcement 
learning environment modelled after the OpenAi Gym API.
From the paper: 
"Deadline Scheduling as Restless Bandits"
'''

import gym
import math
import time
import torch 
import random
import datetime 
import numpy as np
import pandas as pd
from gym import spaces
#from stable_baselines.common.env_checker import check_env #this package throws errors. it's normal. requires python 3.6.

class deadlineSchedulingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    '''
    Custom Gym environment modelled after "deadline scheduling as restless bandits" paper RMAB description.
    The environment represents one position in the N-length queue. 
    '''

    def __init__(self, seed, numEpisodes, episodeLimit, maxDeadline, maxLoad, newJobProb, 
        processingCost, train, batchSize, noiseVar):
        super(deadlineSchedulingEnv, self).__init__()
        self.seed = seed
        self.myRandomPRNG = random.Random(self.seed)
        self.G = np.random.RandomState(self.seed) # create a special PRNG for a class instantiation

        self.observationSize = 2
        self.arm = {0:[1, 1, 1]}  # first: laxity T (D in the paper). Second: load B. Third: deadline d. initalized to all ones

        self.newJobProb = newJobProb
        self.noiseVar = noiseVar
        self.numEpisodes = numEpisodes
        self.currentEpisode = 0
        self.episodeTime = 0
        self.episodeLimit = episodeLimit
        self.train = train
        self.processingCost = processingCost
 
        self.maxDeadline = maxDeadline
        self.maxLoad = maxLoad
        self.batchSize = batchSize
        self.miniBatchCounter = 0
        self.loadIndex = 0

        lowState = np.zeros(self.observationSize, dtype=np.float32)
        highState = np.full(self.observationSize, [self.maxDeadline, self.maxLoad], dtype=np.float32)

        self.action_space = spaces.Discrete(2) 
        self.state_space = spaces.Box(lowState, highState, dtype=np.float32)
        self.createStateTable()
        # gives the added noise value for each state sampled from a Gaussian distribution
        self.noiseVector = self.G.normal(0, np.sqrt(self.noiseVar), np.shape(self.stateArray)[0]*2)

    def _calReward(self, action, state):
        ''' separate function that only retrieves the reward without changing the state.
        For sampling the reward function. '''
        currentState = np.array([self.arm[0][0], self.arm[0][1]], dtype=np.float32)

        if action == 1:
            noise = self.noiseVector[self._findStateIndex(currentState)]
            if (currentState[1] == 0) and (currentState[0] == 0): 
                reward = 0 
            elif (currentState[1] >= 0) and (currentState[0] > 1): 
                reward = (1 - self.processingCost)
                currentState[0] -= 1
                currentState[1] -= 1
                if currentState[1] < 0:
                    reward = 0

            elif (currentState[1] >= 0) and (currentState[0] == 1):  
                reward = ((1 - self.processingCost) - 0.2*(((currentState[1]) - 1)**2)) 
                if (currentState[1] == 0):
                    reward = 0
        elif action == 0:
            noise = self.noiseVector[self._findStateIndex(currentState)+np.shape(self.stateArray)[0]]
            if (currentState[1] == 0)  and (currentState[0] == 0):
                reward = 0
            elif (currentState[1] >= 0) and (currentState[0] > 1): 
                reward = 0
            elif (currentState[1] >= 0) and (currentState[0] == 1):  
                reward =  -0.2*(((currentState[1]))**2)  

        reward = reward + noise*reward
        return reward 

    def _calRewardAndState(self, action):
        ''' function to calculate the reward and next state. '''
        currentState = np.array([self.arm[0][0], self.arm[0][1]], dtype=np.float32)

        if action == 1:
            noise = self.noiseVector[self._findStateIndex(currentState)]
            if (self.arm[0][1] == 0) and (self.arm[0][0] == 0): 
                reward = 0 
                nextState = self._newArrival()
            elif (self.arm[0][1] >= 0) and (self.arm[0][0] > 1): 
                reward = (1 - self.processingCost)
                self.arm[0][0] -= 1
                self.arm[0][1] -= 1
                if self.arm[0][1] < 0:
                    self.arm[0][1] = 0
                    reward = 0
                nextState = np.array([self.arm[0][0], self.arm[0][1]], dtype=np.float32)
            elif (self.arm[0][1] >= 0) and (self.arm[0][0] == 1): 
                reward = ((1 - self.processingCost) - 0.2*(((self.arm[0][1]) - 1)**2)) 
                if (self.arm[0][1] == 0):
                    reward = 0
                self.arm[0][1] = 0
                self.arm[0][0] = 0
                nextState = self._newArrival()

        elif action == 0:
            noise = self.noiseVector[self._findStateIndex(currentState)+np.shape(self.stateArray)[0]]
            if (self.arm[0][1] == 0)  and (self.arm[0][0] == 0):
                reward = 0
                nextState = self._newArrival()
            elif (self.arm[0][1] >= 0) and (self.arm[0][0] > 1): 
                reward = 0
                self.arm[0][0] -= 1
                nextState = np.array([self.arm[0][0], self.arm[0][1]], dtype=np.float32)
            elif (self.arm[0][1] >= 0) and (self.arm[0][0] == 1):  
                reward =  -0.2*(((self.arm[0][1]))**2)  
                self.arm[0][1] = 0
                self.arm[0][0] = 0
                nextState = self._newArrival()

        reward = reward + noise*reward
        return nextState, reward 

    def _findStateIndex(self, state):

        stateLocation = np.where((self.stateArray == state).all(axis=1))[0][0]
        return stateLocation

    def createStateTable(self):
        stateArray = []

        for B in range(self.maxLoad+1): 
            for T in range(self.maxDeadline+1): 
                state = [T,B]
                stateArray.append(state) 

        self.stateArray = np.array(stateArray, dtype=np.float32)

    def step(self, action):
        ''' standard Gym function for taking an action. Provides the next state, reward, and episode termination signal.'''
        assert self.action_space.contains(action)
        assert action in [0,1]
        self.episodeTime += 1

        nextState, reward = self._calRewardAndState(action)
        
        if self.train:
            done = bool(self.episodeTime == self.episodeLimit) 
        else:
            done = False 

        if done:
            self.currentEpisode += 1
            self.episodeTime = 0
            if self.train == False:
                self.currentEpisode = 0

        info = {}

        return nextState, reward, done, info 

    def _newArrival(self): 
        ''' function for new load arrivals during an episode.'''

        job = self.jobList[self.episodeTime-1]
        if job == 1:
            self.arm[0][2] = self.deadline[self.loadIndex]
            self.arm[0][0] = self.timeUntilDeadline[self.loadIndex]
            self.arm[0][1] = self.load[self.loadIndex]
            self.loadIndex += 1

        elif job == 0:
            self.arm[0][2] = 0
            self.arm[0][0] = 0
            self.arm[0][1] = 0
        else:
            print('ERROR. Value not in range...')
            exit(1)

        state = np.array([self.arm[0][0], self.arm[0][1]], dtype=np.float32)

        return state

    def reset(self):
        ''' standard Gym function for reseting the state for a new episode.'''
        self.loadIndex = 0

        if self.miniBatchCounter % self.batchSize == 0:

            self.jobList =  self.G.choice([1,0], p=[self.newJobProb, 1 - self.newJobProb], size=self.episodeLimit)
            
            self.deadline = self.G.randint(1, self.maxDeadline+1, size=self.episodeLimit) 
            self.timeUntilDeadline = self.deadline.copy()
            self.load = self.G.randint(1, self.maxLoad+1, size=self.episodeLimit)

            self.arm[0][2] = self.deadline[0]
            self.arm[0][0] = self.timeUntilDeadline[0]
            self.arm[0][1] = self.load[0]

            self.miniBatchCounter = 0
            
        else:

            self.arm[0][2] = self.deadline[0]
            self.arm[0][0] = self.timeUntilDeadline[0]
            self.arm[0][1] = self.load[0]
        self.episodeTime = 0
        initialState = np.array([self.arm[0][0], self.arm[0][1]], dtype=np.float32)
        self.loadIndex += 1    
        self.miniBatchCounter += 1

        return initialState

#########################################################################################
'''
For environment validation purposes, the below code checks if the nextstate, reward matches
what is expected given a dummy action.
'''
'''
SEED = 50

env = deadlineSchedulingEnv(seed = SEED, numEpisodes = 6, episodeLimit = 20, maxDeadline = 12,
maxLoad=9, newJobProb=0.7, train=True, processingCost = 0.5, batchSize = 1, noiseVar = 0.0)

observation = env.reset()

#check_env(env, warn=True)

x = np.array([1,1,0,0,1])
x = np.tile(x, 10000)
#x = np.random.choice([1,0], size=1000)
n_steps = np.size(x)

start = time.time()
for step in range(n_steps):
    nextState, reward, done, info = env.step(x[step])
    print(f'action: {x[step]} nextstate: {nextState}  reward: {reward} done: {done}')
    print("---------------------------------------------------------")
    if done:
        print(f'Finished episode {env.currentEpisode}/{env.numEpisodes}')
        if env.currentEpisode < env.numEpisodes:
            nextState = env.reset()
        if env.currentEpisode == env.numEpisodes:
            break
  

print(f'-------------------------------------\nDone. Time taken: {time.time() - start:.4f} seconds')
'''