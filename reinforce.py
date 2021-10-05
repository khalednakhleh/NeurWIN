'''
implementation of the REINFORCE algorithm 
with episodic mini-batches.
This REINFORCE algorithm is from:
https://towardsdatascience.com/learning-reinforcement-learning-reinforce-with-pytorch-5e8ad7fc7da0
Please refer to the above link for description.
'''

import os
import gym
import sys
import time
import torch
import random
import numpy as np
import pandas as pd 
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

class reinforceFcnn(nn.Module):
    '''Fully-connected neural network for REINFORCE to train it. 
    There are two different NNs: one for size-aware and deadline scheduling.
    The other NN is for the recovering bandits problem.'''
    def __init__(self, stateDim, actionDim, hidden1, hidden2):
        super(reinforceFcnn, self).__init__()

        self.linear1 = nn.Linear(stateDim, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, actionDim)

        self.printNumParams()

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = F.softmax(x, dim=-1)
        
        return x

    def printNumParams(self): 
        total_params = sum(p.numel() for p in self.parameters())
        total_params_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total number of parameters: {total_params}')
        print(f'Total number of trainable parameters: {total_params_trainable}')


class REINFORCE(object):

    def __init__(self, lr, env, seed, numEpisodes, batchSize, discountFactor, saveDir, 
    	activateArms, episodeSaveInterval, stateDim, actionDim, hidden1, hidden2, numActions):

        #-------------constants-------------
        torch.manual_seed(seed)
        self.seed = seed
        self.myRandomPRNG = random.Random(self.seed)
        self.G = np.random.RandomState(self.seed) # create a special PRNG for a class instantiation
        
        self.numEpisodes = numEpisodes
        self.episodeRanges = np.arange(0, self.numEpisodes+episodeSaveInterval, episodeSaveInterval) 
        self.batchSize = batchSize 
        self.beta = discountFactor
        self.env = env
        self.directory = saveDir
        self.nn = reinforceFcnn(stateDim=stateDim, actionDim=actionDim, hidden1=hidden1, hidden2=hidden2) 
        self.LearningRate = lr  
        self.activateArms = activateArms
        self.numActions = numActions
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.LearningRate)
        #-------------counters-------------
        self.batchCounter = 0
        self.episodeRewards = []
        self.lossFunctionVals = []
        self.plotRewards = []
        self.totalRewards = []

    def discountRewards(self, rewards):
        '''Function for discounting an episode's rewards '''
        r = np.array([self.beta**i * rewards[i] for i in range(len(rewards))])
        r = r[::-1].cumsum()[::-1] 
        result = r - r.mean()
        return result

    
    def learn(self):
        '''Function for initiating the learning process. Gradient ascent steps and environment interactions
        take place here.'''
        self.start = time.time()
        self.currentEpisode = 0
        self.batchCounter = 0
        self.totalTimestep = 0
        batchRewards = []
        batchActions = []
        batchStates = []
        actionSpace = np.arange(self.numActions)

        while self.currentEpisode < self.numEpisodes:
            if self.currentEpisode in self.episodeRanges:
                self.close(self.currentEpisode)

            episodeStart = time.time()
            observation = self.env.reset()
            states = []
            rewards = []
            actions = []
            done = False

            while done == False:
                actionProbs = self.nn.forward(observation).detach().numpy()
                action = self.G.choice(actionSpace, p=actionProbs)
                nextState, reward, done, info = self.env.step(action)

                states.append(observation)
                rewards.append(reward)
                actions.append(action)

                observation = nextState
                self.totalTimestep += 1

                if done:
                    print(f"Finished Episode: {self.currentEpisode+1}")
                    self.totalRewards.append(sum(rewards))
                    batchRewards.extend(self.discountRewards(rewards))
                    batchStates.extend(states)
                    batchActions.extend(actions)
                    self.batchCounter += 1
                    self.currentEpisode += 1

                    
                    if self.batchCounter == self.batchSize:

                        self.optimizer.zero_grad()
                        stateBatchTensor = torch.FloatTensor(batchStates) 
                        rewardBatchTensor = torch.FloatTensor(batchRewards)
                        actionBatchTensor = torch.LongTensor(batchActions)
                        logProb = torch.log(self.nn.forward(stateBatchTensor))

                        selectedLogProbs = rewardBatchTensor * torch.gather(logProb, 1, actionBatchTensor.unsqueeze(1)).squeeze()

                        loss = -selectedLogProbs.mean() 
                        self.lossFunctionVals.append(loss.detach().numpy())

                        loss.backward() 
                        self.optimizer.step() 

                        print(f'did gradient ascent step')
                        
                        batchRewards = []
                        batchActions = []
                        batchStates = []
                        self.batchCounter = 0

        self.end = time.time()
        self.close(self.numEpisodes)
        self.trainingEnding()
        print(f'---------------------------\nDONE. Time taken: {self.end - self.start:.5f} seconds.')
        print(f'total timesteps taken: {self.totalTimestep}')

    def close(self, episode):
        '''Function for saving the NN parameters at defined interval *episodeSaveInterval* '''

        directory=(f'{self.directory}'+f'seed_{self.seed}\
_lr_{self.LearningRate}_batchSize_{self.batchSize}_trainedNumEpisodes_{episode}')
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        torch.save(self.nn.state_dict(), directory+'/trained_model.pt')

    def trainingEnding(self):
        '''Function for saving training information once it is over.''' 
        file = open(self.directory+'trainingInfo.txt', 'w+')
        file.write(f'training time: {self.end - self.start:.5f} seconds\n')   
        file.write(f'training episodes: {self.numEpisodes}\n')  
        file.write(f'Mini-batch size: {self.batchSize}\n')
        file.write(f'Total timesteps: {self.totalTimestep}\n')  
        file.close()



'''
deadline and size aware (4 choose 1)
stateDim = 8, actionDim=4, hidden1=62, hidden2=30

deadline and size aware (10 choose 1)
stateDim=20, actionDim=10, hidden1 =92, hidden2=42


recovering bandits (4 choose 1)
stateDim = 4, actionDim=4, hidden1=64, hidden2=32

recovering bandits (10 choose 1)
stateDim = 10, actionDim=10, hidden1=96, hidden2 =48 

'''