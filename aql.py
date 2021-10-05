

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
#from torchviz import make_dot
import matplotlib.pyplot as plt
import torch.nn.functional as F
from memory import SequentialMemory 
from torch.distributions import Categorical
from operator import itemgetter


class ProposalQNetwork(nn.Module):
    def __init__(self, stateDim, actionDim,hidden1,hidden2):
        super(ProposalQNetwork, self).__init__()

        self.firstLayer = nn.Linear(stateDim, hidden1)
        self.secondLayer = nn.Linear(hidden1, hidden2)
        self.actionLayer = nn.Linear(hidden2, actionDim)
        self.criticLayer = nn.Linear(hidden2, actionDim)


    def forward(self, state):
        state = torch.FloatTensor(state)
        state = F.relu(self.firstLayer(state))
        state = F.relu(self.secondLayer(state))

        actionProbs = F.softmax(self.actionLayer(state), dim=-1)
        
        self.actionDistribution = Categorical(actionProbs)
        
        action = self.actionDistribution.sample().detach().numpy()
        
        stateQValues = self.criticLayer(state)
        
        return action, stateQValues

    def printNumParams(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_params_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total number of parameters: {total_params}')
        print(f'Total number of trainable parameters: {total_params_trainable}')


class AQL(object):
    def __init__(self, lr, env, seed, numEpisodes, discountFactor, stateDim, lamda, epsilon, numActions,
        saveDir, activateArms, episodeSaveInterval, actionDim, iidActionNum, nnActionNum,hidden1,hidden2):

        torch.manual_seed(seed)
        self.seed = seed 
        self.myRandomPRNG = random.Random(self.seed)
        self.G = np.random.RandomState(self.seed)
        self.numEpisodes = numEpisodes
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        self.episodeRanges = np.arange(0, self.numEpisodes+episodeSaveInterval, episodeSaveInterval)
     
        self.beta = discountFactor
        self.env = env
        self.directory = saveDir
        self.stateDim = stateDim
        self.actionDim = actionDim  
        self.nn = ProposalQNetwork(stateDim, actionDim, hidden1, hidden2)
        self.LearningRate = lr 
        self.activateArms = activateArms
        self.numActions = numActions 
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()
        self.memory = SequentialMemory(limit=6000000, ignore_episode_boundaries=False, window_length=1)

        self.M = iidActionNum
        self.N = nnActionNum
        self.epsilon = epsilon 
        self.lamda = torch.tensor(lamda)

    def ActioniidRand(self):
        y = np.zeros(self.numActions - self.activateArms)
        x = np.append(np.ones(self.activateArms), y)
        SampledAction = self.G.permutation(x)
        return SampledAction 

    def iidAction(self):
        iidActions = []
        if self.actionDim  == 100:
            for i in range(self.M):
                iidActions.append(self.ActioniidRand())

            return iidActions
        else:
            for i in range(self.M):
                iidActions.append(np.random.randint(0, self.numActions))

            return np.array([iidActions]).flatten()
    
    def selectAction(self, state):
        
        iidActions = self.iidAction()
        
        nnActions = []
        for i in range(self.N):
            action, qValues = self.nn(state)
            nnActions.append(action)
        nnActions = np.array(nnActions)

        if self.actionDim != 100:

            allActions = np.unique(np.concatenate((iidActions, nnActions)))
            
            asteriskAction = np.argmax(qValues.detach().numpy())
            
            decision = np.random.choice([1,0], p=[self.epsilon, 1-self.epsilon])
            if decision == 1:
                action = np.random.choice(iidActions) 
            else:
                action = asteriskAction

            return action, asteriskAction 
        
        else:
            
            x, qValues = self.nn(state)
            
            indices = (-qValues).argsort()[:self.activateArms]
            notIndices = (-qValues).argsort()[self.activateArms:]
            
            qValues[indices] = 1
            qValues[notIndices] = 0
            asteriskAction = qValues.detach().numpy()

            decision = np.random.choice([1,0], p=[self.epsilon, 1-self.epsilon])
            if decision == 1:
                action = iidActions[self.G.choice(self.M)]
            else:
                action = asteriskAction

            return action, asteriskAction


    def gradientStep(self):
        unrollLength = 300
        self.nn.zero_grad()

        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(unrollLength)
        
        
        loss = 0

        for i in range(len(self.logProb)):
            
            loss += -1*self.logProb[i] - self.lamda * self.entropy[i]

        loss.backward()

        if self.actionDim == 100:
            actionBatch = []

            for x in range(len(action_batch)):

                actionBatch.append(self.G.choice(np.where(action_batch[x] == 1)[0], size=1)[0])

            action_batch = actionBatch

        state_batch = torch.FloatTensor(state_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        action_batch = torch.LongTensor(action_batch).flatten()

        next_state_batch = torch.FloatTensor(next_state_batch)

        x, nextStateQVal = self.nn(next_state_batch)
        x, stateQVal = self.nn(state_batch)

        asteriskActionNextState = torch.argmax(nextStateQVal,dim=1)

        nextStateQVal = torch.gather(nextStateQVal,1, asteriskActionNextState.unsqueeze(1)).squeeze()

        stateQVal = torch.gather(stateQVal,1, action_batch.unsqueeze(1)).squeeze()

        reward_batch = reward_batch.reshape(1,unrollLength)

        stateQVal = stateQVal.reshape(1,unrollLength)

        criticLossVal = self.criterion(reward_batch+self.beta*nextStateQVal, stateQVal)
        criticLossVal.backward()
        
        self.optimizer.step()
        self.logProb = []
        self.entropy = []
        print(f'did gradient descent step')

    def learn(self):
        self.start = time.time() 
        self.currentEpisode = 0 
        self.batchCounter = 0 
        self.totalTimestep = 0 

        self.logProb = []
        self.entropy = []

        while self.currentEpisode < self.numEpisodes:
            if self.currentEpisode in self.episodeRanges:
                self.close(self.currentEpisode)
            
            observation = self.env.reset()

            done = False
            while done == False:

                action, asteriskAction = self.selectAction(observation)

                if self.actionDim == 100: # if 100 arms
                    self.logProb.append(torch.max(self.nn.actionDistribution.log_prob(torch.tensor(asteriskAction))))
                    self.entropy.append(torch.max(self.nn.actionDistribution.entropy()))
                    nextState, reward, done, info = self.env.step(action)
                else:
                    self.logProb.append(self.nn.actionDistribution.log_prob(torch.tensor(asteriskAction)))
                    self.entropy.append(self.nn.actionDistribution.entropy())
                    nextState, reward, done, info = self.env.step(action)
  
                self.memory.append(observation, np.array(action), reward, done)

                observation = nextState
                self.totalTimestep += 1
                self.epsilon -= 1/(80000)
                self.epsilon = max(0, self.epsilon)

                if done:
                    print(f'finished episode: {self.currentEpisode+1}')
                    action, asteriskAction = self.selectAction(observation)
                    self.memory.append(observation, np.array(action), 0., True)

                    self.gradientStep()
                    self.currentEpisode += 1


        self.end = time.time()
        self.close(self.numEpisodes)
        self.trainingEnding()
        print(f'---------------------------\nDONE. Time taken: {self.end - self.start:.5f} seconds.')
        print(f'total timesteps taken: {self.totalTimestep}')

    def close(self, episode):
        '''Function for saving the NN parameters at defined interval *episodeSaveInterval* '''

        directory=(f'{self.directory}'+f'seed_{self.seed}\
_lr_{self.LearningRate}_trainedNumEpisodes_{episode}')
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        torch.save(self.nn.state_dict(), directory+'/trained_model.pt')

    def trainingEnding(self):
        '''Function for saving training information once it is over.''' 
        file = open(self.directory+'trainingInfo.txt', 'w+')
        file.write(f'training time: {self.end - self.start:.5f} seconds\n')   
        file.write(f'training episodes: {self.numEpisodes}\n')
        file.write(f'Total timesteps: {self.totalTimestep}\n')  
        file.close()


'''
deadline and wireless: 

4 choose 1: 56, 32 
10 choose 1: 90, 42
100 choose 25: 512, 196

recovering:
4 choose 1:  64, 32
10 choose 1: 92, 48
100 choose 25: 512, 150
'''