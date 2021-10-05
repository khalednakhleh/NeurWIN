'''
Environment to calculate the Whittle index value of a single restless arm.
Envrionment is a deep reinforcement learning environment modelled after the OpenAi Gym API.
From the paper: 
"Recovering Bandits" NeurIPS 2019.
'''
import GPy
import gym
import math
import time
import torch 
import random
import datetime 
import numpy as np
import pandas as pd
from gym import spaces
from scipy.stats import norm 
import matplotlib.pyplot as plt
#from stable_baselines.common.env_checker import check_env #this package throws errors. it's normal

class recoveringBanditsEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    '''
    Custom Gym environment modelled after "recovering bandits" paper arm's description.
    The environment represents one restless arm.
    '''

    def __init__(self, seed, numEpisodes, episodeLimit, train, 
        batchSize, thetaVals, noiseVar, maxWait = 20):
        super(recoveringBanditsEnv, self).__init__()

        self.seed = seed
        self.myRandomPRNG = random.Random(self.seed)
        self.G = np.random.RandomState(self.seed) # create a special PRNG for a class instantiation

        self.observationSize = 1 # state size
        self.episodeLimit = episodeLimit
        self.train = train 
        self.batchSize = batchSize
        self.maxWait = maxWait
        self.noiseVar = noiseVar
        self.numEpisodes = numEpisodes
        self.episodeTime = 0
        self.currentEpisode = 0
        self.miniBatchCounter = 0

        self.zhist = np.array([], dtype=np.int64).reshape(0, 1)
        self.yhist = np.array([], dtype=np.int64).reshape(0, 1)
        self.kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=5.0)
        self.ucb = 0.0
        self.ts = 0.0
        self.model = None


        self.arm = {0:1} # initial state of the arm.
        
        self.noiseVector = self.G.normal(0, np.sqrt(self.noiseVar), self.maxWait*2)  

        
        val = sum([2**x for x in np.arange(1,self.maxWait+1)])
        self.stateProbs = [2**(x)/(val) for x in np.arange(1,self.maxWait+1)]

        self.theta0 = thetaVals[0]
        self.theta1 = thetaVals[1]
        self.theta2 = thetaVals[2]

        lowState = np.zeros(self.observationSize, dtype=np.float32)
        highState = np.full(self.observationSize, self.maxWait, dtype=np.float32)

        self.action_space = spaces.Discrete(2) 
        self.state_space = spaces.Box(lowState, highState, dtype=np.float32)

    def _calReward(self, action, stateVal):
        '''Function to calculate recovery function's reward based on supplied state.'''
        if action == 1:
            reward = self.theta0 * (1 - np.exp(-1*self.theta1 * stateVal + self.theta2))
            val = int(stateVal-1)
            noise = self.noiseVector[val]
            
        else:
            reward = 0.0
            val = int(stateVal-1 + self.maxWait)
            noise = self.noiseVector[val]

        return reward + (noise*reward)

    def step(self, action):
        ''' Standard Gym function for taking an action. Supplies nextstate, reward, and episode termination signal.'''
        assert self.action_space.contains(action)

        self.episodeTime += 1
        reward = self._calReward(action, self.arm[0])

        self.currentState = self.arm[0]

        if action == 1:
            self.arm[0] = 1
        elif action == 0:
            self.arm[0] = min(self.arm[0]+1, self.maxWait) 

        nextState = np.array([self.arm[0]], dtype=np.float32)

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

    def reset(self):
        ''' Standard Gym function for supplying initial episode state.'''
        if self.train:
            if self.miniBatchCounter % self.batchSize == 0:
                self.arm[0] = self.G.choice(np.arange(1,self.maxWait+1), p=self.stateProbs) 
                initialState = np.array([self.arm[0]], dtype=np.float32)
                self.miniBatchCounter = 0
                self.prevStateVal = self.arm[0]
            else:
                self.arm[0] = self.prevStateVal
                initialState = np.array([self.arm[0]], dtype=np.float32)

            self.miniBatchCounter += 1

        else:
            self.arm[0] = 1 
            initialState = np.array([self.arm[0]], dtype=np.float32)

        return initialState

    def plotRecoveryFunction(self):
        ''' function for plotting the recovery function based on its theta values.'''
        rewards = []
        for i in range(1,self.maxWait+1):
            reward = self.theta0 * (1 - np.exp(-1*self.theta1 * i + self.theta2))
            rewards.append(reward)

        plt.plot(range(1,self.maxWait+1), rewards)
        plt.ylabel('$f_j(z)$')
        plt.xlabel('$z \in \{0, z_{max} = 30\}$')
        plt.grid('on')
        plt.title(f'Theta 0 = {self.theta0}. Theta 1 = {self.theta1}. Theta 2 = {self.theta2}')
        plt.savefig('../plotResults/recovering_gamma_function.png')
        plt.show()
    
    def UpdatePosterior(self, ynew):
        self.zhist = np.vstack((self.zhist, self.currentState))
        self.yhist = np.vstack((self.yhist, ynew))

        if np.shape(self.zhist)[0] >= 50:
            self.zhist = np.array([self.zhist[-50:]], dtype=np.int64).reshape(50,1)
            self.yhist = np.array([self.yhist[-50:]], dtype=np.int64).reshape(50,1)
       
        if self.model is None:
            self.model = GPy.models.GPRegression(X=self.zhist, Y=self.yhist,
                                                 kernel=self.kernel, noise_var=0.01)
        else:
            self.model.set_XY(self.zhist, self.yhist)


##########################################################################################
'''
For environment validation purposes, the below code checks if the nextstate, reward matches
what is expected given a dummy action.
'''
'''
SEED = 100
classATheta = [10., 0.2, 0.0]
classBTheta = [8.5, 0.4, 0.0]
classCTheta = [7., 0.6, 0.0]
classDTheta = [5.5, 0.8, 0.0]
THETAVALS = classATheta
env = recoveringBanditsEnv(seed=SEED, numEpisodes=20, episodeLimit=5, thetaVals = THETAVALS,
train=True, batchSize=5, noiseVar=0.0, maxWait = 20)

observation = env.reset()
#check_env(env, warn=True)

x = np.array([1,0,0,0,1,0])
x = np.tile(x, 10000)
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

'''