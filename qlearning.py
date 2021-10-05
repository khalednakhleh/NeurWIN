'''
Q-learning implementation from the paper:
Towards Q-learning the Whittle Index for Restless Bandits
'''

import random 
import numpy as np 


class qLearningAgent(): 
    def __init__(self, env, stateTable, seed):
        self.counter = 0
        self.learningRate =  0.001 
        self.numStates = np.shape(stateTable)[0]
        self.env = env
        self.beta = 0.99
        self.num_lamda = 100
        self.stateTable = stateTable 
        self.lamdaSet = np.linspace(0, 10, num=self.num_lamda)
        self.seed = seed
        self.myRandomPRNG = random.Random(self.seed)

        self.lamda_qTable = np.zeros((self.num_lamda, self.numStates, 2)) 
        self.init_lamda_index = np.zeros(self.numStates, dtype=np.uint32)

    def _getLamda(self, state):
        '''Get the index value for a given state'''
        self.currentState = self._findStateIndex(state) 
        return self.lamdaSet[self.init_lamda_index[self.currentState]]

    def _takeAction(self, a_1, a_0, flag):

        nextState, reward, done, info = self.env.step(a_1)

        self._updateQTable(self.currentState, nextState, reward, a_1, flag)
    
        self.learningRate = (self.counter)**(-0.5)

        return nextState, reward
 
    def _findStateIndex(self, state):
        
        if(np.size(state) > 1):
            stateLocation = np.where((self.stateTable == state).all(axis=1))[0][0]
        else:
            stateLocation = np.where((self.stateTable == state))[0][0]

        return stateLocation

    def _updateQTable(self, state, new_state, reward, action, flag):
        new_state = self._findStateIndex(new_state)

        self.lamda_qTable[self.init_lamda_index[state],state,action] = self.lamda_qTable[self.init_lamda_index[state], state, action]+self.learningRate*(reward-action*self.lamdaSet[(self.init_lamda_index[state])]+self.beta*
        np.max(self.lamda_qTable[self.init_lamda_index[state],new_state, :]))-self.learningRate*self.lamda_qTable[self.init_lamda_index[state], state, action]
        
        self.counter += 1
        for st in range(len(self.stateTable)):
            ds=np.absolute(self.lamda_qTable[:,st,1]-self.lamda_qTable[:,st,0])
            self.init_lamda_index[st]=np.argmin(ds)
            if flag == 1:
                self.init_lamda_index[st] = self.myRandomPRNG.choice(range(len(self.lamdaSet)))