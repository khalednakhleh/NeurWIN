'''
Gym deadline scheduling environment for fixed n arms.
To be used with the REINFORCE algorithm.
Can be used with any OpenAI baseline agent.
This environment acts as a wrapper environment for N deadlineScheduling enviornments.
'''


import gym
import time
import random
import itertools 
import numpy as np 
import pandas as pd 
import scipy.special
from gym import spaces
from deadlineSchedulingEnv import deadlineSchedulingEnv
#from stable_baselines.common.env_checker import check_env #this package throws errors. it's normal

class deadlineSchedulingMultipleArmsEnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, seed, numEpisodes, batchSize, train, numArms, processingCost, 
        maxDeadline, maxLoad, newJobProb, episodeLimit, scheduleArms, noiseVar):
        super(deadlineSchedulingMultipleArmsEnv, self).__init__()

        self.seed = seed
        self.myRandomPRNG = random.Random(self.seed)
        self.G = np.random.RandomState(self.seed) # create a special PRNG for a class instantiation

        self.time = 0
        self.numEpisodes = numEpisodes
        self.episodeTime = 0
        self.currentEpisode = 0
        self.numArms = numArms
        self.episodeLimit = episodeLimit
        self.batchSize = batchSize
        self.state = []
        self.envs = {}
        self.noiseVar = noiseVar
        self.newJobProb = newJobProb
        self.train = train
        self.processingCost = processingCost
        self.maxDeadline = maxDeadline
        self.maxLoad = maxLoad
        self.scheduleArms = scheduleArms
        self.observationSize = self.numArms*2
 
        self._createActionTable()

        maxState = np.tile([self.maxDeadline, self.maxLoad], self.numArms)
        lowState = np.zeros(self.observationSize, dtype=np.float32)
        highState = np.full(self.observationSize, maxState, dtype=np.float32)

        #self.action_space = spaces.Discrete(len(self.actionTable))
        self.state_space = spaces.Box(lowState, highState, dtype=np.float32)

        self.envSeeds = self.G.randint(0, 10000, size=self.numArms)
        self._setTheArms()

    def _createActionTable(self):
        '''function that creates a mapping of actions to take. Will be mapped with the action taken from the agent.'''
        if self.numArms <= 20:
            self.actionTable = np.zeros(int(scipy.special.binom(self.numArms, self.scheduleArms)))
            n = int(self.numArms)
            self.actionTable  = list(itertools.product([0, 1], repeat=n))
            self.actionTable = [x for x in self.actionTable if not sum(x) != self.scheduleArms]
            self.action_space = spaces.Discrete(len(self.actionTable))
        else:
            self.actionTable = None 

    def _calReward(self, action):
        '''Function to calculate recovery function's reward based on supplied state.'''
        
        if self.actionTable != None:
        
            actionVector = self.actionTable[action]
        else:
            actionVector = action 
        cumReward = 0
        state = []
        
        envCounter = 0
        for i in range(len(actionVector)):
            if actionVector[i] == 1:
                nextState, reward, done, info = self.envs[envCounter].step(1)
                state.append(nextState[0])
                state.append(nextState[1])
                cumReward += reward
            elif actionVector[i] == 0:
                nextState, reward, done, info = self.envs[envCounter].step(0)
                state.append(nextState[0])
                state.append(nextState[1])
                cumReward += reward
            envCounter += 1

        state = np.array(state, dtype=np.float32)

        return state, cumReward

    def step(self, action):
        ''' Standard Gym function for taking an action. Supplies nextstate, reward, and episode termination signal.'''
        
        self.time += 1
        self.episodeTime += 1

        nextState, reward = self._calReward(action)
        done = bool(self.episodeTime == self.episodeLimit)

        if done: 
            self.currentEpisode += 1
            self.episodeTime = 0

        info = {}
        
        return nextState, reward, done, info

    def _setTheArms(self):
        ''' function that sets the N arms for training'''
        for i in range(self.numArms):

            self.envs[i] = deadlineSchedulingEnv(seed=self.envSeeds[i], numEpisodes=1, episodeLimit=self.episodeLimit, 
                maxDeadline=self.maxDeadline, maxLoad=self.maxLoad, newJobProb=self.newJobProb,
                processingCost=self.processingCost, train=False, batchSize=self.batchSize, noiseVar=self.noiseVar)

    def reset(self):
        ''' Standard Gym function for supplying initial episode state.
        Episodes in the same mini-batch have the same trajectory for valid return comparison.'''
        self.state = []

        for i in self.envs:
            vals = self.envs[i].reset()
            val1 = vals[0]
            val2 = vals[1]
            self.state.append(val1)
            self.state.append(val2)

        self.state = np.array(self.state, dtype=np.float32)
        
        return self.state

##########################################################################################
'''
For environment validation purposes, the below code checks if the nextstate, reward matches
what is expected given a dummy action.
'''
'''
SEED = 30
x = np.array([0,1,3,2,1])
x = np.tile(x, 1000)
n_steps = np.size(x)
noiseVar = 0.0
howManyToActivate = 1

env = deadlineSchedulingMultipleArmsEnv(seed=SEED, numEpisodes=10, batchSize=5, train=True, numArms=4, processingCost=0.5, 
        maxDeadline=12, maxLoad=9, newJobProb=0.7, episodeLimit=10, scheduleArms=howManyToActivate, noiseVar=noiseVar)

observation = env.reset()


start = time.time()
for step in range(n_steps):
    #print(x[step])
    nextState, reward, done, info = env.step(x[step])
    print(f'action: {x[step]} nextstate: {nextState} reward: {reward} done: {done}')
    print("---------------------------------------------------------")
    if done:
        print(f'Finished episode {env.currentEpisode}/{env.numEpisodes}')
        if env.currentEpisode < env.numEpisodes:
            nextState = env.reset()
        if env.currentEpisode == env.numEpisodes:
            break


print(f'-------------------------------------\nDone. Time taken: {time.time() - start:.4f} seconds')
print(env.actionTable)
print(len(env.actionTable))

'''