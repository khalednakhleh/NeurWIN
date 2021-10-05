'''
Main file for running training experiments for both NeurWIN and REINFORCE.
For selecting the training case, uncomment its portion in the file.
'''

import os
import sys
import gym
import time
import torch
import random
import numpy as np
from aql import AQL
import pandas as pd
from reinforce import REINFORCE
from neurwin import NEURWIN, fcnn
sys.path.insert(0,'envs/')
from sizeAwareIndexEnv import sizeAwareIndexEnv
from recoveringBanditsEnv import recoveringBanditsEnv
from deadlineSchedulingEnv import deadlineSchedulingEnv
from sizeAwareIndexMultipleArmsEnv import sizeAwareIndexMultipleArmsEnv
from recoveringBanditsMultipleArmsEnv import recoveringBanditsMultipleArmsEnv
from deadlineSchedulingMultipleArmsEnv import deadlineSchedulingMultipleArmsEnv


###########################PARAMETERS########################################
STATESIZE = 2
SEED = 50
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

sigmoidParam = 5     # sigmoid function sensitivity parameter. 1 for deadline. 5 for recovering. 0.75 for size-aware
BATCHSIZE = 5        # gradient ascent step every n episodes
learningRate = 1e-03   
numEpisodes = 30000
discountFactor = 0.99

TRAIN = True
EPISODELIMIT = 300 # 300 timesteps for training
noiseVar = 0.0
#####################################################
# for size-aware index's cases
CASE = 1
CLASSVAL = 1

if CASE == 1 and CLASSVAL == 1:
    HOLDINGCOST = 1
    GOODTRANS = 33600 
    BADTRANS = 8400
    GOODPROB = 0.75
    LOAD = 1000000

elif CASE == 1 and CLASSVAL == 2:
    HOLDINGCOST = 1
    GOODTRANS = 33600 
    BADTRANS = 8400
    GOODPROB = 0.1
    LOAD = 1000000

else:
    print(f'entered case not in list. Exiting...')
    exit(1)

########################################----TRAINING SETTINGS----#########################################################
########################################## NEURWIN DEADLINE SCHEDULING ###################################################
'''
if noiseVar > 0: 
    deadlineDirectory = (f'trainResults/neurwin/deadline_env/noise_{noiseVar}_version/')
else:
    deadlineDirectory = (f'trainResults/neurwin/deadline_env/')

deadlineEnv = deadlineSchedulingEnv(seed=SEED, numEpisodes=numEpisodes, episodeLimit=EPISODELIMIT, maxDeadline=12,
maxLoad=9, newJobProb=0.7, processingCost=0.5, train=TRAIN, batchSize=BATCHSIZE, noiseVar=noiseVar)

agent = NEURWIN(stateSize=STATESIZE,lr=learningRate, env=deadlineEnv, sigmoidParam=sigmoidParam, numEpisodes=numEpisodes, noiseVar=noiseVar,
seed=SEED, batchSize=BATCHSIZE, discountFactor=discountFactor, saveDir = deadlineDirectory, episodeSaveInterval=5)
agent.learn()
'''
########################################## NEURWIN WIRELESS SCHEDULING #################################################
'''
if noiseVar > 0:
    sizeAwareDirectory = (f'trainResults/neurwin/size_aware_env/noise_{noiseVar}_version/case_{CASE}/class_{CLASSVAL}/')
else:
    sizeAwareDirectory = (f'trainResults/neurwin/size_aware_env/case_{CASE}/class_{CLASSVAL}/')

sizeAwareEnv = sizeAwareIndexEnv(numEpisodes=numEpisodes, HOLDINGCOST=HOLDINGCOST, seed=SEED, Training=TRAIN, r1=BADTRANS,
r2=GOODTRANS, q=GOODPROB, case=CASE, classVal=CLASSVAL, noiseVar = noiseVar, 
load=LOAD, batchSize = BATCHSIZE, maxLoad = LOAD, episodeLimit=EPISODELIMIT, fixedSizeMDP=False)

agent = NEURWIN(stateSize=STATESIZE,lr=learningRate, env=sizeAwareEnv, sigmoidParam=sigmoidParam, numEpisodes=numEpisodes, noiseVar=noiseVar,
seed=SEED, batchSize=BATCHSIZE, discountFactor=discountFactor, saveDir = sizeAwareDirectory, episodeSaveInterval=100)
agent.learn()
'''
###################################### NEURWIN RECOVERING BANDITS SCHEDULING ##############################################
'''
maxWait = 20 # maximum time before refreshing the arm
STATESIZE = 1
CASE = 'A'  # A,B,C,D different recovery functions

if CASE == 'A':
	THETA = [10., 0.2, 0.0]
elif CASE == 'B':
	THETA = [8.5, 0.4, 0.0]
elif CASE == 'C':
	THETA = [7., 0.6, 0.0]
elif CASE == 'D':
	THETA = [5.5, 0.8, 0.0]


if noiseVar > 0:
    recoveringDirectory = (f'trainResults/neurwin/recovering_bandits_env/noise_{noiseVar}_version/recovery_function_{CASE}/')
else:
    recoveringDirectory = (f'trainResults/neurwin/recovering_bandits_env/recovery_function_{CASE}/')

os.makedirs(recoveringDirectory)
file = open(recoveringDirectory+'used_parameters.txt', 'w+')
file.write(f'Theta0, Theta1, Theta2: {THETA}\n')
file.write(f'max wait for recovery function: {maxWait}\n')
file.close()

print(f'selected theta: {THETA}')
recoveringEnv = recoveringBanditsEnv(seed=SEED, numEpisodes=numEpisodes, episodeLimit=EPISODELIMIT, train=TRAIN, 
batchSize=BATCHSIZE,thetaVals=THETA, noiseVar=noiseVar, maxWait = maxWait)

agent = NEURWIN(stateSize=STATESIZE,lr=learningRate, env=recoveringEnv, noiseVar=noiseVar,
sigmoidParam=sigmoidParam, numEpisodes=numEpisodes,seed=SEED, batchSize=BATCHSIZE, 
discountFactor=discountFactor, saveDir = recoveringDirectory,episodeSaveInterval=100)
agent.learn()
'''
###################################### REINFORCE SIZE-AWARE SCHEDULING ##############################################
'''
numArms = 10
stateDim = numArms*2
actionDim = numArms
hidden1 = 92
hidden2 = 42
class1Arms = class2Arms = int(numArms / 2)
activateArms = 1
numActions = numArms

reinforceSizeAwareDirectory = (f'trainResults/reinforce/size_aware_env/case_{CASE}/arms_{numArms}_schedule_{activateArms}/')


sizeAwareIndexMultipleArmsEnv = sizeAwareIndexMultipleArmsEnv(seed=SEED, numEpisodes=numEpisodes,train=TRAIN, noiseVar=0,
batchSize = BATCHSIZE, class1Arms=class1Arms, class2Arms=class2Arms, numArms=numArms, scheduleArms=activateArms, case=CASE, episodeLimit=EPISODELIMIT)

reinforceAgent = REINFORCE(lr=learningRate, env=sizeAwareIndexMultipleArmsEnv, seed=SEED, activateArms = activateArms,
numEpisodes=numEpisodes, batchSize=BATCHSIZE, discountFactor=discountFactor, saveDir = reinforceSizeAwareDirectory,episodeSaveInterval=100,
stateDim=stateDim, actionDim=actionDim, hidden1=hidden1, hidden2=hidden2, numActions=numActions)


reinforceAgent.learn()
'''
###################################### REINFORCE DEADLINE SCHEDULING ##############################################
'''
numArms = 10
stateDim = numArms*2
actionDim = numArms
hidden1 = 92
hidden2 = 42
numActions = numArms

newJobProb = 0.7
activateArms = 1
PROCESSINGCOST = 0.5
MAXDEADLINE = 12
MAXLOAD = 9

reinforceDeadlineDirectory = (f'trainResults/reinforce/deadline_env/arms_{numArms}_schedule_{activateArms}/')

deadlineMultipleArmsEnv = deadlineSchedulingMultipleArmsEnv(seed=SEED, numEpisodes=numEpisodes, batchSize=BATCHSIZE, 
train=True, numArms=numArms, processingCost=PROCESSINGCOST, maxDeadline=MAXDEADLINE, 
maxLoad=MAXLOAD, newJobProb=newJobProb, episodeLimit=EPISODELIMIT, scheduleArms=activateArms, noiseVar=noiseVar)

reinforceAgent = REINFORCE(lr=learningRate, env=deadlineMultipleArmsEnv, seed=SEED, activateArms = activateArms,
numEpisodes=numEpisodes, batchSize=BATCHSIZE, discountFactor=discountFactor, saveDir = reinforceDeadlineDirectory,episodeSaveInterval=5,
stateDim=stateDim, actionDim=actionDim, hidden1=hidden1, hidden2=hidden2, numActions = numActions)


reinforceAgent.learn()
'''
################################# REINFORCE RECOVERING BANDITS SCHEDULING #########################################
'''
NUMARMS = 10

stateDim = NUMARMS
actionDim = NUMARMS
hidden1 = 96
hidden2 = 48
SCHEDULEARMS = 1
numActions = NUMARMS
MAXWAIT = 20

reinforceRecoveringDirectory = (f'trainResults/reinforce/recovering_env/arms_{NUMARMS}_schedule_{SCHEDULEARMS}/')

recoveringMultipleArmsEnv = recoveringBanditsMultipleArmsEnv(seed=SEED, numEpisodes=numEpisodes, batchSize=BATCHSIZE,
train = True, numArms=NUMARMS, scheduleArms=SCHEDULEARMS, noiseVar=noiseVar, maxWait=MAXWAIT, episodeLimit=EPISODELIMIT)

reinforceAgent = REINFORCE(lr=learningRate, env=recoveringMultipleArmsEnv, seed=SEED, activateArms = SCHEDULEARMS,
numEpisodes=numEpisodes, batchSize=BATCHSIZE, discountFactor=discountFactor, saveDir = reinforceRecoveringDirectory,episodeSaveInterval=100,
stateDim=stateDim, actionDim=actionDim, hidden1=hidden1, hidden2=hidden2, numActions=numActions)


reinforceAgent.learn()
'''

################################# AQL DEADLINE SCHEDULING #########################################
'''

newJobProb = 0.7
numArms = 10
activateArms = 1
PROCESSINGCOST = 0.5
MAXDEADLINE = 12
MAXLOAD = 9
STATEDIM = numArms*2
LAMDA = 0.8
ACTIONDIM = numArms
numActions = numArms
epsilon = 1
M = 10 # iid actions
N = 10 # actor actions
hidden1 = 90
hidden2 = 42

aqlDeadlineSaveDir = (f'trainResults/aql/deadline_env/arms_{numArms}_schedule_{activateArms}/')

deadlineMultipleArmsEnv = deadlineSchedulingMultipleArmsEnv(seed=SEED, numEpisodes=numEpisodes, batchSize=numEpisodes, 
train=True, numArms=numArms, processingCost=PROCESSINGCOST, maxDeadline=MAXDEADLINE, maxLoad=MAXLOAD, newJobProb=newJobProb, 
episodeLimit=EPISODELIMIT, scheduleArms=activateArms, noiseVar=noiseVar)

aqlAgent = AQL(lr=learningRate, epsilon=epsilon, env=deadlineMultipleArmsEnv, seed=SEED, numEpisodes=numEpisodes, numActions=numActions,
discountFactor=discountFactor, stateDim=STATEDIM, lamda=LAMDA, saveDir=aqlDeadlineSaveDir, activateArms=activateArms, 
episodeSaveInterval=5, actionDim=ACTIONDIM, iidActionNum=M, nnActionNum=N, hidden1=hidden1 ,hidden2=hidden2)

aqlAgent.learn()
'''
################################# AQL RECOVERING SCHEDULING #########################################
'''
activateArms = 1
MAXWAIT = 20

numArms = 10

STATEDIM = numArms
LAMDA = 0.8
ACTIONDIM = numArms
numActions = numArms
epsilon = 1
M = 10 # iid actions
N = 10 # actor actions
hidden1 = 92
hidden2 = 48

aqlRecoveringSaveDir = (f'trainResults/aql/recovering_env/arms_{numArms}_schedule_{activateArms}/')

recoveringMultipleArmsEnv = recoveringBanditsMultipleArmsEnv(seed=SEED, numEpisodes=numEpisodes, batchSize=BATCHSIZE,
train = True, numArms=numArms, scheduleArms=activateArms, noiseVar=noiseVar, maxWait=MAXWAIT, episodeLimit=EPISODELIMIT)

aqlAgent = AQL(lr=learningRate, epsilon=epsilon, env=recoveringMultipleArmsEnv, seed=SEED, numEpisodes=numEpisodes, numActions=numActions,
discountFactor=discountFactor, stateDim=STATEDIM, lamda=LAMDA, saveDir=aqlRecoveringSaveDir, activateArms=activateArms, 
episodeSaveInterval=100, actionDim=ACTIONDIM, iidActionNum=M, nnActionNum=N, hidden1=hidden1, hidden2=hidden2)

aqlAgent.learn()

'''
################################# AQL WIRELESS SHCEDULING ########################################
'''
numArms = 10
STATEDIM = numArms*2
ACTIONDIM = numArms
hidden1 = 90
hidden2 = 42
class1Arms = class2Arms = int(numArms / 2)
activateArms = 1
numActions = numArms
epsilon = 1
LAMDA = 0.8
M = 10
N = 10

aqlSizeAwareDirectory = (f'trainResults/aql/size_aware_env/case_{CASE}/arms_{numArms}_schedule_{activateArms}/')

sizeAwareIndexMultipleArmsEnv = sizeAwareIndexMultipleArmsEnv(seed=SEED, numEpisodes=numEpisodes,train=TRAIN, noiseVar=0,
batchSize = BATCHSIZE, class1Arms=class1Arms, class2Arms=class2Arms, numArms=numArms, scheduleArms=activateArms, case=CASE, episodeLimit=EPISODELIMIT)

aglAgent = AQL(lr=learningRate, epsilon=epsilon, env=sizeAwareIndexMultipleArmsEnv, seed=SEED, numEpisodes=numEpisodes, numActions=numActions,
discountFactor=discountFactor, stateDim=STATEDIM, lamda=LAMDA, saveDir=aqlSizeAwareDirectory, activateArms=activateArms, 
episodeSaveInterval=100, actionDim=ACTIONDIM, iidActionNum=M, nnActionNum=N, hidden1=hidden1, hidden2=hidden2)

aglAgent.learn()
'''