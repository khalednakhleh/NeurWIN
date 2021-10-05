'''
Gym scheduling environment for n arms.
To be used with the REINFORCE algorithm.
Can be used with any OpenAI baseline agent.
This environment acts as a wrapper environment for N recoveringBandits Env.
'''


import gym
import time
import random
import itertools 
import numpy as np 
import pandas as pd 
import scipy.special
from gym import spaces
from recoveringBanditsEnv import recoveringBanditsEnv
#from stable_baselines.common.env_checker import check_env #this package throws errors. it's normal

class recoveringBanditsMultipleArmsEnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, seed, numEpisodes, batchSize, train, numArms, 
        scheduleArms, noiseVar, maxWait, episodeLimit):
        super(recoveringBanditsMultipleArmsEnv, self).__init__()

        self.seed = seed        
        self.myRandomPRNG = random.Random(self.seed)
        self.G = np.random.RandomState(self.seed) # create a special PRNG for a class instantiation

        self.time = 0
        self.numEpisodes = numEpisodes
        self.episodeLimit = episodeLimit
        self.observationSize = numArms 
        self.numArms = numArms
        self.batchSize = batchSize
        self.train = train  
        self.maxWait = maxWait 
        self.noiseVar = noiseVar
        self.episodeTime = 0
        self.currentEpisode = 0
        self.scheduleArms = scheduleArms
        self._createActionTable()
        self.envs = {}

        lowState = np.ones(self.observationSize, dtype=np.float32)
        highState = np.array(np.tile(self.maxWait, self.observationSize), dtype=np.float32)

        
        self.state_space = spaces.Box(lowState, highState, dtype=np.float32)

        self.envSeeds = self.G.randint(0, 10000, size=self.numArms)
        classATheta = [10., 0.2, 0.0]
        classBTheta = [8.5, 0.4, 0.0]
        classCTheta = [7., 0.6, 0.0]
        classDTheta = [5.5, 0.8, 0.0]
        self.THETA = []

        for i in range(self.numArms): 
            self.THETA.append(classATheta)
            self.THETA.append(classBTheta)
            self.THETA.append(classCTheta)
            self.THETA.append(classDTheta)

        self._setTheArms()
    
    def _createActionTable(self):
        ''' function that creates a mapping of actions to take. Will be mapped with the action taken from the agent.'''
        if self.numArms <= 20:
            self.actionTable = np.zeros(int(scipy.special.binom(self.numArms, self.scheduleArms)))
            n = int(self.numArms)
            self.actionTable  = list(itertools.product([0, 1], repeat=n))
            self.actionTable = [x for x in self.actionTable if not sum(x) != self.scheduleArms]
            self.action_space = spaces.Discrete(len(self.actionTable))
        else:
            self.actionTable = None 

    def _setTheArms(self):
        ''' function that sets the N arms for training'''

        for i in range(self.numArms):
            
            self.envs[i] = recoveringBanditsEnv(seed=self.envSeeds[i], numEpisodes=1, episodeLimit=self.episodeLimit, train=True, 
        batchSize=self.batchSize, thetaVals=self.THETA[i], noiseVar=self.noiseVar, maxWait = self.maxWait)
            
    def _calculateReward(self, action):
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
                cumReward += reward # only obtain reward from activated arm(s)
            elif actionVector[i] == 0:
                nextState, reward, done, info = self.envs[envCounter].step(0)
                state.append(nextState[0])
            envCounter += 1

        state = np.array(state, dtype=np.float32)

        return state, cumReward

    def step(self, action):
        ''' Standard Gym function for taking an action. Supplies nextstate, reward, and episode termination signal.'''
        #assert self.action_space.contains(action)
        self.time += 1
        self.episodeTime += 1

        nextState, reward = self._calculateReward(action)

        done = bool(self.episodeTime == self.episodeLimit)

        if done:
            self.currentEpisode += 1
            self.episodeTime = 0

        info = {}

        return nextState, reward, done, info

    def reset(self):
        ''' Standard Gym function for supplying initial episode state.
        Episodes in the same mini-batch have the same trajectory for valid return comparison.'''
        self.state = []

        for i in self.envs:
            val = self.envs[i].reset()
            self.state.append(val[0])

        self.state = np.array(self.state, dtype=np.float32)
        
        return self.state

##########################################################################################
'''
For environment validation purposes, the below code checks if the nextstate, reward matches
what is expected given a dummy action.
'''
'''
SEED = 25
x = np.array([0,1,2,3])
x = np.tile(x, 1000)
n_steps = np.size(x)

howManyToActivate = 1

env = recoveringBanditsMultipleArmsEnv(seed=SEED, numEpisodes=1, batchSize=5, train=True, numArms=10, 
        scheduleArms=howManyToActivate, noiseVar=0.0, maxWait=20, episodeLimit=10)
#env = deadlineMultipleArmsEnv(seed=SEED, numEpisodes=10, batchSize=5, train=True, numArms=4, processingCost=0.5, 
#        maxDeadline=12, maxLoad=9, newJobProb=0.7, episodeLimit=1, scheduleArms=howManyToActivate)

observation = env.reset()

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


print(env.state_space.shape[0])
print(env.actionTable)
'''