'''
Whittle index Q-learning for one arm.
'''

import random
import numpy as np 



class WIBQL():
    def __init__(self, numEpisodes, episodeLimit, numArms, env, stateTable):
        
        self.env = env 
        self.beta = 0.99
        self.numStates = np.shape(stateTable)[0]
        self.stateTable = stateTable
        self.timeHorizon =  episodeLimit*numEpisodes
        self.indices = np.zeros((self.numStates)) 
        
        self.numArms = numArms
        
        self.qTable = np.zeros((self.numStates, self.numStates, 2)) 
        self.indexCounter = 0
        self.stateActionCounters = np.zeros((self.numStates, 2), dtype=int) 
    
    def _getIndexBnStepSize(self):
        n = self.indexCounter + 1
        modulo = 1 if np.mod(n, self.numArms) != 0 else 0
        stepSize = (1 / (1 + np.ceil((n*np.log10(n))/500)))*modulo
        return stepSize

    def _getLamda(self, state):
        self.currentState = self._findStateIndex(state)
        return self.indices[self.currentState]

    def _takeAction(self, action):
        nextState, reward, done, info = self.env.step(action)

        self.updateQTable(currentState=self.currentState, nextState=nextState, action=action, reward=reward)

        return nextState

    def _findStateIndex(self, state):
        
        if(np.size(state) > 1): 
            stateLocation = np.where((self.stateTable == state).all(axis=1))[0][0]
        else:
            stateLocation = np.where((self.stateTable == state))[0][0]

        return stateLocation

    def updateIndex(self): 
        b_n = self._getIndexBnStepSize()
        for i in range(self.numStates):
            self.indices[i] = self.indices[i] + b_n*(self.qTable[i, i, 1] - self.qTable[i, i, 0])

        self.indexCounter += 1
    
    def _getAn(self, stateActionCounter):
        
        return 1/(np.ceil((stateActionCounter+1) / 500))

    def updateQTable(self, currentState, nextState, action, reward):

        nextState = self._findStateIndex(nextState)
        stepSize = self._getAn(self.stateActionCounters[currentState, action])
        
        for k in range(self.numStates):
            fQFunction = (1 / (2*self.numStates)) * np.sum(self.qTable[:,k, 1] + self.qTable[:,k, 0])
            armReward = ((1 - action)*(reward + self.indices[k])) + (action*reward)
            qVal = self.beta*np.max(self.qTable[nextState, k, :]) - fQFunction - self.qTable[currentState, k, action]
            self.qTable[currentState, k, action] = self.qTable[currentState, k, action] + stepSize*(armReward + qVal)
        
        self.stateActionCounters[currentState, action] += 1

