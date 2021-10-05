'''
Gym scheduling environment for n arms.
To be used with the REINFORCE algorithm.
Can also be used with any OpenAI baseline agent.
This environment acts as a wrapper environment for N sizeAwareIndex enviornments.
'''

import gym
import random
import itertools 
import numpy as np
import pandas as pd
import scipy.special
from gym import spaces
from sizeAwareIndexEnv import sizeAwareIndexEnv
#from stable_baselines.common.env_checker import check_env #this package throws errors. it's normal

class sizeAwareIndexMultipleArmsEnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, seed, numEpisodes, batchSize, train, noiseVar,
    class1Arms, class2Arms, numArms, scheduleArms, case, episodeLimit):
        super(sizeAwareIndexMultipleArmsEnv, self).__init__()
        self.seed = seed
        self.myRandomPRNG = random.Random(self.seed)
        self.G = np.random.RandomState(self.seed) # create a special PRNG for a class instantiation

        assert(case in [1,2])
        assert(class1Arms+class2Arms == numArms)
        self.time = 0
        self.numEpisodes = numEpisodes
        self.episodeTime = 0
        self.currentEpisode = 0
        self.numArms = numArms
        self.class1Arms = class1Arms
        self.class2Arms = class2Arms
        self.noiseVar = noiseVar

        self.scheduleArms = scheduleArms
        self.case = case
        self.episodeLimit = episodeLimit
        self.batchSize = batchSize
        self.state = []
        self.envs = {}
        self.q1 = 0.75 
        self.q2 = 0.1  
        self.train = train

        if case == 1:
            self.goodTrans1 = 33600.0
            self.badTrans1 = 8400.0
            self.goodTrans2 = 33600.0
            self.badTrans2 = 8400.0
            self.holdingCost1 = 1.0
            self.holdingCost2 = 1.0
            self.loadClass1 = 1000000.0
            self.loadClass2 = 1000000.0
        elif case == 2:
            self.goodTrans1 = 33600.0
            self.badTrans1 = 8400.0
            self.goodTrans2 = 33600.0
            self.badTrans2 = 8400.0
            self.holdingCost1 = 5.0
            self.holdingCost2 = 1.0
            self.loadClass1 = 1000000.0
            self.loadClass2 = 1000000.0
        else:
            print('Case not in case list. exiting...')
            exit(1)
        
        self._createActionTable()
        self.envSeeds = self.G.randint(0, 10000, size=self.numArms)

        self.observationSize = numArms*2 
        self.low = np.zeros(self.observationSize, dtype=np.float32)
        self.high = np.full(self.observationSize, 1.0, dtype=np.float32)

        
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.state_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.miniBatchCounter = 0
        self.prevLoads = []
        self.prevChannelState = []
        self._setTheArms()
    
    def _createActionTable(self):
        '''function to create a mapping of actions to take. Will be mapped with the action taken from the agent.
        Set of permissible actions is mapped to binomial factor N choose M'''
        if self.numArms <= 20:
            self.actionTable = np.zeros(int(scipy.special.binom(self.numArms, self.scheduleArms)))
            n = int(self.numArms)
            self.actionTable  = list(itertools.product([0, 1], repeat=n))
            self.actionTable = [x for x in self.actionTable if not sum(x) != self.scheduleArms]
            self.action_space = spaces.Discrete(len(self.actionTable))
        else:
            self.actionTable = None 

    def _calReward(self, action):
        '''function to calculate the total reward from taking action x. Reward is collected from all arms.'''

        if self.actionTable != None:
            actionVector = self.actionTable[action]
        else:
            actionVector = action 

        cumulativeReward = 0
        state = []
        remaining = 0 
        counter = []

        for i in self.envs:
            if actionVector[i] == 1:
                nextState, reward, done, info = self.envs[i].step(1)
                state.append(nextState[0])
                state.append(nextState[1])
                remaining += nextState[0]
                cumulativeReward += reward
                if (nextState[0] == 0):
                    counter.append(self.envs[i].classVal)

            elif actionVector[i] == 0:
                nextState, reward, done, info = self.envs[i].step(0)
                state.append(nextState[0])
                state.append(nextState[1])
                remaining += nextState[0]
                cumulativeReward += reward
                if nextState[0] == 0:
                    counter.append(self.envs[i].classVal)

        state = np.array(state, dtype=np.float32)

        if len(counter) == self.numArms:
            val = counter[-1]
            if val == 1:
                cumulativeReward += -1*self.holdingCost1
            else:
                cumulativeReward += -1*self.holdingCost2

        return state, cumulativeReward, remaining

    def step(self, action):
        ''' standard Gym function for taking an action. Provides the next state, reward, and episode termination signal.'''
        #assert self.action_space.contains(action)
        self.time += 1
        self.episodeTime += 1
 
        nextState, reward, remainingLoad = self._calReward(action)
        done = bool((remainingLoad == 0)) or (self.episodeTime == self.episodeLimit)

        if done:
            self.currentEpisode += 1
            self.episodeTime = 0
            for x in self.envs:
                self.envs[x].currentEpisode += 1
                self.envs[x].episodeTime = 0

        info = {}
        return nextState, reward, done, info 

    def _setTheArms(self):
        ''' initialization of all arms that define the MDP. The learning algorithm is trained on this set of arms.'''
        num1 = self.class1Arms
        num2 = self.class2Arms

        goodTrans1 = self.goodTrans1
        badTrans1 = self.badTrans1
        goodTrans2 = self.goodTrans2
        badTrans2 = self.badTrans2
        loadClass1 = self.loadClass1
        loadClass2 = self.loadClass2
        
        for i in range(self.numArms):
            if num1 != 0:  
                self.envs[i] = sizeAwareIndexEnv(seed=self.envSeeds[i], numEpisodes=self.numEpisodes, HOLDINGCOST=self.holdingCost1, Training=self.train,
                r1=badTrans1, r2=goodTrans1, q=self.q1, case=self.case, classVal=1, noiseVar=self.noiseVar,
                batchSize=self.batchSize, load=loadClass1, maxLoad=loadClass1, episodeLimit=self.episodeLimit, fixedSizeMDP=True)
                num1 -= 1
                
            elif num2 != 0:
                self.envs[i] = sizeAwareIndexEnv(seed=self.envSeeds[i], numEpisodes=self.numEpisodes, HOLDINGCOST=self.holdingCost2, Training=self.train,
                r1=badTrans2, r2=goodTrans2, q=self.q2, case=self.case, classVal=2, noiseVar=self.noiseVar,
                batchSize=self.batchSize, load=loadClass2, maxLoad=loadClass2, episodeLimit=self.episodeLimit, fixedSizeMDP=True)
                num2 -= 1
                
            i += 1

    def reset(self):
        ''' Standard Gym function for supplying initial episode state.'''
        self.state = []
        for i in self.envs:
            vals = self.envs[i].reset()
            val1 = vals[0]
            val2 = vals[1]
            self.state.append(val1)
            self.state.append(val2)
        
        self.state = np.array(self.state, dtype=np.float32)
        
        return self.state

##############################################################################################
'''
For environment validation purposes, the below code checks if the nextstate, reward matches
what is expected given a dummy action.
'''

'''
numEpisodes = 1
SEED = 90
training = True
class1Arms = 5
class2Arms = 5
numArms = 10
case = 1
BATCHSIZE = 5
howManyToSchedule = 1
EPISODELIMIT = 100
noiseVar = 0.0

env = sizeAwareIndexMultipleArmsEnv(numEpisodes=numEpisodes, seed=SEED, batchSize = BATCHSIZE, train=training, noiseVar=noiseVar,
    class1Arms=class1Arms, class2Arms=class2Arms, numArms=numArms, scheduleArms = howManyToSchedule, case=case, episodeLimit = EPISODELIMIT)
#check_env(env, warn=True)
observation = env.reset()

x = np.array([0,2,5,7]) # dummy actions for N = 4, M = 1.
x = np.tile(x, 200)

n_steps = np.size(x)
for step in range(n_steps):
    nextState, reward, done, info = env.step(x[step])
    print(f'action: {x[step]} next state: {nextState}  reward: {reward} done: {done}')
    print("---------------------------------------------------------")
    if done:
        print(f'Finished episode {env.currentEpisode}/{env.numEpisodes}')
        if env.currentEpisode < env.numEpisodes:
            nextState = env.reset()
        if env.currentEpisode == env.numEpisodes:
            break

print(env.actionTable)
print(len(env.actionTable))
'''