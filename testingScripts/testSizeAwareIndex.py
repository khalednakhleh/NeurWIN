"""
testing a trained model for scheduling,
and comparing its performance with the 
size-aware whittle index heuristic.
To test a control policy, uncomment the code corresponding to an algorithm.
"""

import os 
import torch
import random
import operator
import itertools 
import numpy as np
import pandas as pd 
import scipy.special
import sys
sys.path.insert(0,'../')
from neurwin import fcnn 
from qlearning import qLearningAgent
import matplotlib.pyplot as plt
from aql import ProposalQNetwork
from envs.sizeAwareIndexEnv import sizeAwareIndexEnv
from  reinforce import reinforceFcnn, REINFORCE 

###########################-CONSTANT VALUES-########################################
STATESIZE = 2
numEpisodes = 1 
SEED = 30
filesSeed = 50  
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
################################-PARAMETERS-########################################
CASE = 1 
SCHEDULE = 1        
REINFORCELR = 0.001  
BATCHSIZE = 5
ARMS = 4           
BETA = 0.99
TIMELIMIT = 300
numClass1 = 2
numClass2 = 2
EPISODESEND = 1
EPISODERANGE = 1000
RUNS = 50
noiseVar = 0
NOISY = False 
noiseVal = 0.0 
assert(numClass1+numClass2 == ARMS)

if CASE == 1:
    HOLDINGCOST1 = 1
    HOLDINGCOST2 = 1
    MAXLOAD1 = 1000000
    MAXLOAD2 = 1000000
    TESTLOAD1 = np.random.randint(1, MAXLOAD1, size=ARMS)
    TESTLOAD2 = np.random.randint(1, MAXLOAD2, size=ARMS)

    GOODTRANS1 = 33600
    BADTRANS1 = 8400

    GOODTRANS2 = 33600
    BADTRANS2 = 8400

    GOODPROB1 = 0.75
    GOODPROB2 = 0.1

if CASE == 2: 
    HOLDINGCOST1 = 5
    HOLDINGCOST2 = 1
    MAXLOAD1 = 1000000
    MAXLOAD2 = 1000000
    TESTLOAD1 = np.random.randint(1, MAXLOAD1, size=ARMS)
    TESTLOAD2 = np.random.randint(1, MAXLOAD2, size=ARMS)
    GOODTRANS1 = 33600
    BADTRANS1 = 8400

    GOODTRANS2 = 33600
    BADTRANS2 = 8400

    GOODPROB1 = 0.5
    GOODPROB2 = 0.5


if NOISY:
    directory = (f'../testResults/size_aware_env/noisy_results/noise_val_{noiseVal}/case_{CASE}/')
    WINNMODEL1DIR = (f'../trainResults/neurwin/size_aware_env/noise_{noiseVal}_version/case_{CASE}/class_1/') 
    WINNMODEL2DIR = (f'../trainResults/neurwin/size_aware_env/noise_{noiseVal}_version/case_{CASE}/class_2/') 
    if not os.path.exists(directory):
        os.makedirs(directory)   

else:
    directory = (f'../testResults/size_aware_env/case_{CASE}/')
    WINNMODEL1DIR = (f'../trainResults/neurwin/size_aware_env/case_{CASE}/class_1/') 
    WINNMODEL2DIR = (f'../trainResults/neurwin/size_aware_env/case_{CASE}/class_2/')

    if not os.path.exists(directory):
        os.makedirs(directory)

readMeFileName = (f'{directory}'+'readme.txt')
readMeFile = open(readMeFileName, 'a')
readMeFile.write(f'\nSelected case: {CASE}\nNumber of arms: {ARMS} \nNumber of class 1 arms: {numClass1+numClass2}')
readMeFile.close()


REINFORCEDIR = (f'../trainResults/reinforce/size_aware_env/case_{CASE}/arms_{ARMS}_schedule_{SCHEDULE}/')
WOLPDIR = (f'../trainResults/wolp_ddpg/size_aware_env/arms_{ARMS}_schedule_{SCHEDULE}/')
AQLDIR = (f'../trainResults/aql/size_aware_env/case_{CASE}/arms_{ARMS}_schedule_{SCHEDULE}/')
##########################-- TESTING FUNCTIONS --#########################################

def calculateSecondaryIndex():
    global goodEnvs, goodIndex
    for i in goodEnvs:
        nuem = envs[i].holdingCost * envs[i].goodTransVal
        denom =  envs[i].arm[0][0]
        goodIndex[i] = nuem / denom
    
def getSelectionSizeAware(goodIndex, badIndex):
    result = []
    copyGoodIndex = goodIndex.copy()
    copyBadIndex = badIndex.copy()
    if len(copyGoodIndex) + len(copyBadIndex) == SCHEDULE:
        armsToActivate = SCHEDULE - len(copyGoodIndex)
    else:
        armsToActivate = len(copyBadIndex)

    armsToActivate = min(SCHEDULE, len(copyGoodIndex) + len(copyBadIndex))

    for i in range(armsToActivate):
        if len(copyGoodIndex) != 0:
            result.append(max(copyGoodIndex.items(), key=operator.itemgetter(1))[0])
            del copyGoodIndex[result[-1]]   
        else:
            result.append(max(copyBadIndex.items(), key=operator.itemgetter(1))[0])
            del copyBadIndex[result[-1]]                 

    return result

def getSelection(index):
    result = []
    copyIndex = index.copy()

    if len(copyIndex) < SCHEDULE:
        for i in range(len(copyIndex)):
            result.append(max(copyIndex.items(), key=operator.itemgetter(1))[0])
            del copyIndex[result[i]]    
    else:   
        for i in range(SCHEDULE):
            result.append(max(copyIndex.items(), key=operator.itemgetter(1))[0])
            del copyIndex[result[i]]
 
    choice = result
    return choice 

def calculatePrimaryIndex():
    global badEnvs, badIndex
    for i in badEnvs:
        nuem = envs[i].holdingCost 
        denom = envs[i].goodProb*((envs[i].goodTransVal/envs[i].badTransVal) - 1)
        badIndex[i] = nuem / denom

def initialize(): 
    global numClass1, numClass2, TESTLOAD1, TESTLOAD2, envSeeds, envs
    num1 = numClass1
    num2 = numClass2 
    load1Index = 0
    load2Index = 0

    for i in range(ARMS):
        if num1 != 0:
            env = sizeAwareIndexEnv(numEpisodes=numEpisodes, HOLDINGCOST=HOLDINGCOST1, seed=envSeeds[i], Training=False,
            r1=BADTRANS1, r2=GOODTRANS1, q=GOODPROB1, case=CASE, classVal=1, load=TESTLOAD1[load1Index], noiseVar = noiseVar,
            maxLoad = MAXLOAD1, batchSize=EPISODESEND, episodeLimit=1000000, fixedSizeMDP=False)
            load1Index += 1
            num1 -= 1
        elif num2 != 0:
            env = sizeAwareIndexEnv(numEpisodes=numEpisodes, HOLDINGCOST=HOLDINGCOST2, seed=envSeeds[i], Training=False,
        r1=BADTRANS2, r2=GOODTRANS2, q=GOODPROB2, case=CASE, classVal=2, load=TESTLOAD2[load2Index], noiseVar = noiseVar,
        maxLoad = MAXLOAD2, batchSize=EPISODESEND, episodeLimit=1000000, fixedSizeMDP=False)
            load2Index += 1
            num2 -= 1

        envs[i] = env

  
def initializeAgents():
    global MODELNAME1, MODELNAME2, agents
    num1 = numClass1
    num2 = numClass2
    for i in range(ARMS):
        if num1 != 0:
            agent = fcnn(stateSize=STATESIZE)
            agent.load_state_dict(torch.load(MODELNAME1))
            agent.eval()
            agents[i] = agent
            num1 -= 1
        
        elif num2 != 0:
            agent = fcnn(stateSize=STATESIZE)
            agent.load_state_dict(torch.load(MODELNAME2))
            agent.eval()
            agents[i] = agent   
            num2 -= 1
                 
def resetEnvs():
    global states, envs
    for key in envs:
        state = envs[key].reset()
        states[key] = state


def calculateIndexNeuralNetwork():
    global indexNN, states
    for key in agents:
        indexNN[key] = agents[key].forward(states[key]).detach().numpy()[0]
    
    choice = getSelection(indexNN)

    indexNN = {}
    return choice

def selectArmSizeAwareIndex():
    global goodEnvs, badEnvs, goodIndex, badIndex, time, states
    for key in envs:
        if envs[key].channelState[time] == 1:
            goodEnvs.append(key)
        else:
            badEnvs.append(key)

    calculateSecondaryIndex() 
    calculatePrimaryIndex()  
    arms = getSelectionSizeAware(goodIndex, badIndex)

    goodEnvs = []
    badEnvs = []
    goodIndex = {}
    badIndex = {}

    return arms

def takeActionAndRecordNN(arms):
    global rewards, time, states, envs

    cumReward = 0 
    for arm in arms:
        nextState, reward, done, info = envs[arm].step(1)
        cumReward += reward 
        states[arm] = nextState

        if done:
            del envs[arm]
            del agents[arm]

    for key in envs:
        if key in arms:
            pass
        else:
            nextState, reward, done, info = envs[key].step(0)
            states[key] = nextState
            cumReward += reward

    rewards.append((BETA**time)*cumReward)

def takeActionAndRecordRewardSizeAwareIndex(arms):
    global rewards, time, envs, states

    cumReward = 0
    
    for arm in arms:
        nextState, reward, done, info = envs[arm].step(1)
        cumReward += reward 
        states[arm] = nextState

        if done:
            del envs[arm] 

    for key in envs:
        if key in arms:
            pass
        else:
            nextState, reward, done, info = envs[key].step(0)
            states[key] = nextState
            cumReward += reward

    rewards.append((BETA**time)*cumReward)

def getActionTableLength():
    scheduleArms = SCHEDULE
    actionTable = np.zeros(int(scipy.special.binom(ARMS, scheduleArms)))
    n = int(ARMS)
    actionTable  = list(itertools.product([0, 1], repeat=n))
    actionTable = [x for x in actionTable if not sum(x) != scheduleArms]
    
    return actionTable


def REINFORCETakeActionAndRecordReward():
    global rewards, state, reinforceAgent, envs, actionTable

    cumReward = 0
    stateVals = []

    action_probs = reinforceAgent.forward(state).detach().numpy()
    G = np.random.RandomState()
    action = G.choice(np.arange(len(actionTable)), p=action_probs)
    actionVector = actionTable[action]


    for i in range(len(actionVector)):
        if actionVector[i] == 1:
            nextState, reward, done, info = envs[i].step(1)
            stateVals.append(nextState[0])
            stateVals.append(nextState[1])
            if nextState[0] != 0.:
                cumReward += reward
        else:
            nextState, reward, done, info = envs[i].step(0)
            stateVals.append(nextState[0])
            stateVals.append(nextState[1])
            if nextState[0] != 0.:
                cumReward += reward 

    state = stateVals
    state = np.array(state, dtype=np.float32)

    rewards.append((BETA**time)*cumReward)

def resetREINFORCEEnvs():
    global envs, state
    for key in envs:
        vals = envs[key].reset()
        val1 = vals[0]
        val2 = vals[1]
        state.append(val1)
        state.append(val2)
    state = np.array(state, dtype=np.float32)


##########################TESTING-STEP######################################
#######################  SIZE-AWARE INDEX ##################################
'''
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

cumReward = []
for i in range(RUNS):
    envSeeds = np.random.randint(0, 10000, size=ARMS)

    time = 0
    envs = {}
    rewards = [] 
    goodEnvs = []
    badEnvs = []
    #index = {}
    goodIndex = {}
    badIndex = {}
    agents = {}
    states = {}
    indexNN = {}
    LOADS = []

    initialize()
    resetEnvs()


    while (time < TIMELIMIT):
        arms = selectArmSizeAwareIndex()
        takeActionAndRecordRewardSizeAwareIndex(arms)
        ###############################################
        time += 1
        if len(envs) == 0: 
            break

    total_reward = (np.cumsum(rewards))[-1]
    cumReward.append(total_reward)
    
    print(f'Finished size aware index value for run {i+1} scheduling {SCHEDULE} arms')

data = {'run': range(RUNS), 'cumulative_reward':cumReward}
df = pd.DataFrame(data=data)
sizeAwareFileName = (f'{directory}'+f'sizeAwareIndexResults_arms_{ARMS}_schedule_{SCHEDULE}_arms.csv')
df.to_csv(sizeAwareFileName, index=False)
'''
################################### NEURWIN TESTING ##########################################
'''
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

for i in range(RUNS):
    envSeeds = np.random.randint(0, 10000, size=ARMS)
    ALLEPISODES = np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE) 
    zeroEnvs = {}
    total_reward = []
    for x in ALLEPISODES:

        EPISODESTRAINED = x
        MODELNAME1 = WINNMODEL1DIR+(f'seed_{filesSeed}_lr_0.001_batchSize_{BATCHSIZE}_trainedNumEpisodes_{EPISODESTRAINED}/trained_model.pt')
        MODELNAME2 = WINNMODEL2DIR+(f'seed_{filesSeed}_lr_0.001_batchSize_{BATCHSIZE}_trainedNumEpisodes_{EPISODESTRAINED}/trained_model.pt')
        nnFileName = directory+(f'nnIndexResults_arms_{ARMS}_batchSize_{BATCHSIZE}_run_{i}_schedule_{SCHEDULE}_arms.csv')
        ############################## NN INDEX TEST ####################################

        time = 0
        envs = {}
        rewards = []
        goodEnvs = []
        badEnvs = []
        index = {}
        agents = {}
        states = {}
        indexNN = {}

        initialize()
        initializeAgents()

        resetEnvs()
        

        while (time < TIMELIMIT):
            
            arms = calculateIndexNeuralNetwork()
            takeActionAndRecordNN(arms)
            ###############################################
            time += 1
            if len(envs) == 0: 
                break
       
        total_reward.append((np.cumsum(rewards))[-1])
        print(f'finished NN scheduling for episode {x}')
    data = {'episode': np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE), 'cumulative_reward':total_reward}
    df = pd.DataFrame(data=data)
    df.to_csv(nnFileName, index=False)
    print(f'finished NN scheduling for run {i+1}')
'''
############################### REINFORCE TESTING ########################################
'''
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

actionTable = getActionTableLength()

hidden1 = 92
hidden2 = 42
for i in range(RUNS):
    envSeeds = np.random.randint(0, 10000, size=ARMS)
    ALLEPISODES = np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE) 
    zeroEnvs = {}
    total_reward = []
    remainingLoad = 0

    for x in ALLEPISODES:

        EPISODESTRAINED = x
        REINFORCEMODELDIR = REINFORCEDIR+(f'seed_{filesSeed}_lr_{REINFORCELR}_batchSize_{BATCHSIZE}_trainedNumEpisodes_{EPISODESTRAINED}/trained_model.pt')
        reinforceFileName = directory+(f'reinforceResults_arms_{ARMS}_batchSize_{BATCHSIZE}_lr_{REINFORCELR}_run_{i}_schedule_{SCHEDULE}.csv')
        time = 0
       
        envs = {}
        rewards = []
        state = []
        initialize()

        resetREINFORCEEnvs()
        
        reinforceAgent = reinforceFcnn(stateDim=ARMS*2, actionDim=ARMS, hidden1=hidden1, hidden2=hidden2) 
        reinforceAgent.load_state_dict(torch.load(REINFORCEMODELDIR))
        reinforceAgent.eval()

        while (time < TIMELIMIT):
            REINFORCETakeActionAndRecordReward()
            time += 1
            for b in envs:
                remainingLoad += envs[b].arm[0][0]
            if remainingLoad == 0:
                break
            remainingLoad = 0

        total_reward.append((np.cumsum(rewards))[-1])
        print(f'finished REINFORCE scheduling for episode {x}. rewards: {total_reward[-1]}')

    data = {'episode': np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE), 'cumulative_reward':total_reward}
    df = pd.DataFrame(data=data)
    df.to_csv(reinforceFileName, index=False)
    print(f'finished REINFOCE scheduling for run {i+1}')

'''

########################## AQL TESTING ###############################

def AQLTakeActionAndRecordReward():
    global rewards, state, aqlAgent, envs, actionTable

    cumReward = 0
    stateVals = []

    action, qVals = aqlAgent.forward(state)

    
    if len(envs) == 100:
        indices = (-qVals).argsort()[:SCHEDULE]
        notIndices = (-qVals).argsort()[SCHEDULE:]
        
        qVals[indices] = 1
        qVals[notIndices] = 0
        actionVector = qVals.detach().numpy()
    else:
        
        actionVector = actionTable[action]
        
    
    for i in range(len(actionVector)):
        if actionVector[i] == 1:
            nextState, reward, done, info = envs[i].step(1)
            stateVals.append(nextState[0])
            stateVals.append(nextState[1])
            if nextState[0] != 0.:
                cumReward += reward
        else:
            nextState, reward, done, info = envs[i].step(0)
            stateVals.append(nextState[0])
            stateVals.append(nextState[1])
            if nextState[0] != 0.:
                cumReward += reward 

    state = stateVals
    state = np.array(state, dtype=np.float32)

    rewards.append((BETA**time)*cumReward)

'''
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#actionTable = getActionTableLength()

DIM1 = 512
DIM2 = 196

for i in range(RUNS):
    envSeeds = np.random.randint(0, 10000, size=ARMS)
    ALLEPISODES = np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE) 
    total_reward = []
    for x in ALLEPISODES:

        EPISODESTRAINED = x
        AQLMODELDIR = AQLDIR+(f'seed_{filesSeed}_lr_0.001_trainedNumEpisodes_{EPISODESTRAINED}/trained_model.pt')
        aqlFileName = directory+(f'aqlResults_arms_{ARMS}_run_{i}_schedule_{SCHEDULE}.csv')
        time = 0
        
        envs = {}
        rewards = []
        state = []
        initialize()

        resetREINFORCEEnvs()
        aqlAgent = ProposalQNetwork(ARMS*2, ARMS, DIM1, DIM2)
        aqlAgent.load_state_dict(torch.load(AQLMODELDIR))
        aqlAgent.eval()
        
        while True:
            AQLTakeActionAndRecordReward()
            time += 1
            if time == TIMELIMIT:
                break

        total_reward.append((np.cumsum(rewards))[-1])
        print(f'finished AQL for trained episodes: {x}. rewards : {total_reward[-1]}')

    data = {'episode': np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE), 'cumulative_reward':total_reward}
    df = pd.DataFrame(data=data)
    df.to_csv(aqlFileName, index=False)
    print(f'finished AQL scheduling for run {i+1}')
'''

