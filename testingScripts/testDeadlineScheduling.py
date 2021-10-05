'''
testing a trained model for scheduling,
and comparing its performance with the 
Deadline Scheduling problem.
To test a control policy, uncomment the code corresponding to an algorithm.
'''

import os 
import torch
import random
import operator
import itertools
import numpy as np 
import pandas as pd 
import scipy.special
import time as timer
import sys
sys.path.insert(0,'../')
from neurwin import fcnn 
import matplotlib.pyplot as plt
from reinforce import reinforceFcnn
from qlearning import qLearningAgent
from wibql import WIBQL
from aql import ProposalQNetwork
from envs.deadlineSchedulingEnv import deadlineSchedulingEnv

###########################-CONSTANT VALUES-########################################
STATESIZE = 2
filesSeed = 50
TIMELIMIT = 300
SEED = 30
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
################################-PARAMETERS-########################################
numEpisodes = 1
ARMS = 100         
SCHEDULE = 25     
PROCESSINGCOST = 0.5
BATCHSIZE = 5
EPISODESEND = 2000
EPISODERANGE = 20
BETA = 0.99
REINFORCELR = 0.001
RUNS = 50

NOISY = False 
noiseVal = 0.0
if NOISY:
    directory = (f'../testResults/deadline_env/noisy_results/noise_val_{noiseVal}/')
    WINNMODELDIR = (f'../trainResults/neurwin/deadline_env/noise_{noiseVal}_version/')
    if not os.path.exists(directory):
        os.makedirs(directory)
else:
    directory = (f'../testResults/deadline_env/')
    WINNMODELDIR = (f'../trainResults/neurwin/deadline_env/')
    REINFORCEDIR = (f'../trainResults/reinforce/deadline_env/arms_{ARMS}_schedule_{SCHEDULE}/')
    AQLDIR = (f'../trainresults/aql/deadline_env/arms_{ARMS}_schedule_{SCHEDULE}/')
    WOLPMODELDIR = (f'../trainResults/wolp_ddpg/deadline_env/arms_{ARMS}_schedule_{SCHEDULE}/')
    if not os.path.exists(directory):
        os.makedirs(directory)  

readMeFileName = (f'{directory}'+'readme.txt')
readMeFile = open(readMeFileName, 'a')
readMeFile.write(f'\nNumber of arms: {ARMS}\n-----------')
readMeFile.close()

############################################################################
def initialize(): 

    global envSeeds, envs
    jobProbs = 0.7
    
    for i in range(ARMS):
        env = deadlineSchedulingEnv(seed=envSeeds[i], numEpisodes=numEpisodes, episodeLimit=TIMELIMIT, maxDeadline=12,
maxLoad=9, newJobProb=jobProbs, processingCost=PROCESSINGCOST, train=False, batchSize=EPISODESEND, noiseVar=0)
        envs[i] = env

def initializeNN(): 

    global envSeeds, envs, agents
    jobProbs = 0.7

    for i in range(ARMS):
        env = deadlineSchedulingEnv(seed=envSeeds[i], numEpisodes=numEpisodes, episodeLimit=TIMELIMIT, maxDeadline=12,
maxLoad=9, newJobProb=jobProbs, processingCost=PROCESSINGCOST, train=False, batchSize=EPISODESEND, noiseVar=0)
        agent = fcnn(stateSize=STATESIZE)
        agent.load_state_dict(torch.load(MODELNAME))
        agent.eval()
        envs[i] = env
        agents[i] = agent

def resetEnvs():
    global states, envs
    for key in envs:
        state = envs[key].reset()
        states[key] = state

def getSelection(index):

    result = []
    copyIndex = index.copy()

    for i in range(SCHEDULE):
        result.append(max(copyIndex.items(), key=operator.itemgetter(1))[0])
        del copyIndex[result[i]]

    choice = result

    return choice 

def calculateIndexNeuralNetwork():
    global indexNN, states
    for key in envs:
        indexNN[key] = agents[key].forward(states[key]).detach().numpy()[0]
    #print(states)
    choice = getSelection(indexNN)
    #print(choice)
    indexNN = {}
    return choice

def takeActionAndRecordNN(arms):
    global rewards, time, states, envs

    cumReward = 0 
    for arm in arms:
        nextState, reward, done, info = envs[arm].step(1)
        cumReward += reward 
        states[arm] = nextState

    for key in envs:
        if key in arms:
            pass
        else:
            nextState, reward, done, info = envs[key].step(0)
            states[key] = nextState
            cumReward += reward

    rewards.append((BETA**time)*cumReward)


def selectDeadlineIndex():
    global states, PROCESSINGCOST, index
    for key in states:
        if states[key][1] == 0:  
            index[key] = 0
        elif (states[key][1] >= 1) and (states[key][1] <= (states[key][0] - 1)): 
            index[key] = 1 - PROCESSINGCOST
        elif (states[key][0] <= states[key][1]): 
            firstVal = (BETA**(states[key][0]-1))*(0.2*(states[key][1] - states[key][0] + 1)**2) 
            secondVal = (BETA**(states[key][0]-1))*(0.2*(states[key][1] - states[key][0])**2) 
            index[key] = firstVal - secondVal + 1 - PROCESSINGCOST
    
    choice = getSelection(index)

    return choice

def takeActionAndRecordDeadlineIndex(arms):
    global rewards, time, envs, states

    cumReward = 0
    
    for arm in arms:
        nextState, reward, done, info = envs[arm].step(1)
        states[arm][0] = nextState[0]
        states[arm][1] = nextState[1]
        cumReward += reward 

    for key in envs:
        if key in arms:
            pass
        else:
            nextState, reward, done, info = envs[key].step(0)
            cumReward += reward        
            states[key][0] = nextState[0]
            states[key][1] = nextState[1]

    rewards.append((BETA**time)*cumReward)

def getActionTableLength():
    global SCHEDULE
    scheduleArms = SCHEDULE
    actionTable = np.zeros(int(scipy.special.binom(ARMS, scheduleArms)))
    n = int(ARMS)
    actionTable  = list(itertools.product([0, 1], repeat=n))
    actionTable = [x for x in actionTable if not sum(x) != scheduleArms]
    
    return actionTable

def resetREINFORCEEnvs():
    global envs, state
    for key in envs:
        vals = envs[key].reset()
        val1 = vals[0]
        val2 = vals[1]
        state.append(val1)
        state.append(val2)
    state = np.array(state, dtype=np.float32)

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
            cumReward += reward
        else:
            nextState, reward, done, info = envs[i].step(0)
            stateVals.append(nextState[0])
            stateVals.append(nextState[1])
            cumReward += reward 

    state = stateVals
    state = np.array(state, dtype=np.float32)

    rewards.append((BETA**time)*cumReward)

################################## Q-learning functions ###############################################


def createStateTable():

    global stateArray

    for B in range(9+1): 
        for T in range(12+1): 
            state = [T,B]
            stateArray.append(state) 

    stateArray = np.array(stateArray, dtype=np.float32)


def initializeQLearningTraining():
    global trainingEnvSeeds, trainingEnvs

    jobProbs = 0.7

    for i in range(ARMS):
        env = deadlineSchedulingEnv(seed=trainingEnvSeeds[i], numEpisodes=EPISODESEND, episodeLimit=TIMELIMIT, maxDeadline=12,
maxLoad=9, newJobProb=jobProbs, processingCost=PROCESSINGCOST, train=True, batchSize=BATCHSIZE, noiseVar=0)
        trainingEnvs[i] = env

def createQLearningAgents():
    global agents, trainingEnvs, stateArray, trainingEnvSeeds

    for key in trainingEnvs:
        agents[key] = qLearningAgent(trainingEnvs[key], stateArray, trainingEnvSeeds[key])

def updateAgentEnvs():
    global agents, envs 
    for key in envs:
        agents[key].env = envs[key]

def qLearningChooseArmsTraining():

    global states, trainingEnvs, agents, time, epsilon

    for key in trainingEnvs:
        index[key] = agents[key]._getLamda(states[key])

    choice = getSelection(index)
    originalActions = [0]*len(trainingEnvs) 

    for i in trainingEnvs:
        if i in choice: 
            originalActions[i] = 1 
        else:
            originalActions[i] = 0
    lambda_flag = 0
    decision = np.random.choice([1,0], p=[epsilon, 1-epsilon])

    if decision == 0:
        madeActions = originalActions.copy() 
    else:
        lambda_flag = 1
        madeActions = random.sample(originalActions, len(originalActions))


    return madeActions, originalActions, lambda_flag

def TakeActionAndRecordQLearning(selectedActions, originalActions, flag):
    global rewards, time, trainingEnvs

    cumReward = 0
    for i in trainingEnvs:
        nextState, reward = agents[i]._takeAction(selectedActions[i], originalActions[i], flag)
        states[i] = nextState

def TakeActionAndRecordWiqbl(selectedActions):
    global rewards, time, trainingEnvs
    
    cumReward = 0
    for i in trainingEnvs:
        nextState = agents[i]._takeAction(selectedActions[i])
        states[i] = nextState


def resetQLearningEnvs():
    global states, trainingEnvs
    for key in trainingEnvs:
        state = trainingEnvs[key].reset()
        states[key] = state


def createQLearningAgentsTesting():
    global agents, trained_vals, envs, stateArray, envSeeds
    for key in envs:
        agents[key] = qLearningAgent(envs[key], stateArray, envSeeds[key])
        agents[key].init_lamda_index = trained_vals[key]

def qLearningTestSelectArms():
    global states, envs, index, agents
    
    for key in envs:
        index[key] = agents[key]._getLamda(states[key])

    choice = getSelection(index)

    return choice

###########################-    TESTING SETTINGS    -######################################
#########################- DEADLINE CLOSED-FORM INDEX SCHEDULING -#########################

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
    index = {}
    states = {}

    initialize()
    resetEnvs()
    

    while True:

        arm = selectDeadlineIndex()
        takeActionAndRecordDeadlineIndex(arm)
        ###############################################
        time += 1
        if time == TIMELIMIT:
            break

    total_reward = (np.cumsum(rewards))[-1]
    cumReward.append(total_reward)
    print(f'finished closed-form deadline index value for run {i+1}. total reward: {cumReward[-1]}')

data = {'run': range(RUNS), 'cumulative_reward':cumReward}
df = pd.DataFrame(data=data)
deadlineFileName = (f'{directory}'+f'deadlineIndexResults_arms_{ARMS}_timeLimit_{TIMELIMIT}_schedule_{SCHEDULE}.csv')
df.to_csv(deadlineFileName, index=False)
'''
##########################- NEURWIN INDEX SCHEDULING -###############################
'''
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

for i in range(RUNS):
    envSeeds = np.random.randint(0, 10000, size=ARMS)
    
    ALLEPISODES = np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE) 
    total_reward = []
    for x in ALLEPISODES:
        EPISODESTRAINED = x
        MODELNAME = WINNMODELDIR+(f'seed_{filesSeed}_lr_0.001_batchSize_{BATCHSIZE}_trainedNumEpisodes_{EPISODESTRAINED}/trained_model.pt')
        nnFileName = directory+(f'nnIndexResults_arms_{ARMS}_batchSize_{BATCHSIZE}_run_{i}_timeLimit_{TIMELIMIT}_schedule_{SCHEDULE}.csv')

        time = 0
        envs = {}
        rewards = [] 
        index = {}
        agents = {}
        states = {}
        indexNN = {}
        whittleIndex = []
        initializeNN()

        resetEnvs()
        
        
        while True:
            
            arm = calculateIndexNeuralNetwork()
            takeActionAndRecordNN(arm)
            ###############################################
            time += 1
            if time == TIMELIMIT: 
                break
            
        total_reward.append((np.cumsum(rewards))[-1])
        print(f'finish NeurWIN for trained episodes: {x}. rewards: {total_reward[-1]}')
    data = {'episode': np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE), 'cumulative_reward':total_reward}
    df = pd.DataFrame(data=data)
    df.to_csv(nnFileName, index=False)
    print(f'finished NeurWIN scheduling for run {i+1}')

'''
##############################- REINFORCE SCHEDULING -######################################
'''
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

actionTable = getActionTableLength()
hidden1=92
hidden2=42

for i in range(RUNS):
    envSeeds = np.random.randint(0, 10000, size=ARMS)
    ALLEPISODES = np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE) 
    zeroEnvs = {}
    total_reward = []
    for x in ALLEPISODES:

        EPISODESTRAINED = x
        REINFORCEMODELDIR = REINFORCEDIR+(f'seed_{filesSeed}_lr_0.001_batchSize_{BATCHSIZE}_trainedNumEpisodes_{EPISODESTRAINED}/trained_model.pt')
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
        
        while True:
            REINFORCETakeActionAndRecordReward()
            time += 1
            if time == TIMELIMIT:
                break

        total_reward.append((np.cumsum(rewards))[-1])
        print(f'finished REINFORCE for trained episodes: {x}. rewards : {total_reward[-1]}')

    data = {'episode': np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE), 'cumulative_reward':total_reward}
    df = pd.DataFrame(data=data)
    df.to_csv(reinforceFileName, index=False)
    print(f'finished REINFOCE scheduling for run {i+1}')

'''
############################### QWIC training ########################################
'''
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

 
episodeRange = np.arange(0, (EPISODESEND + EPISODERANGE), EPISODERANGE) 
trainingEnvs = {}
states = {}
new_states = {}
agents = {}
index = {} 
stateArray = []
trainingEnvSeeds = np.random.randint(0, 10000, size=ARMS)

createStateTable()
initializeQLearningTraining()
createQLearningAgents()
time = 0
epsilon = 1  
start = timer.time()
for episode in range(EPISODESEND):
    resetQLearningEnvs()
    print(f'current episode: {episode+1}')

    if (episode == 0):
        index_vals = []
        for key in agents: 
            index_vals.append(agents[key].init_lamda_index)
        index_vals = np.array(index_vals)
        
        saveDir = (f'../trainResults/qLearning/deadline_env/arms_{ARMS}_schedule_{SCHEDULE}/episode_{episode}')
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        np.save(saveDir, index_vals) 
    for step in range(TIMELIMIT):
        selectedActions, originalActions, lambda_flag = qLearningChooseArmsTraining()
        TakeActionAndRecordQLearning(selectedActions, originalActions, lambda_flag)
        time += 1
        epsilon = min(1,2*(time)**(-0.5))

    if (episode+1 in episodeRange) or (episode+1 == EPISODESEND):
        print(f'saving for episode {episode+1}')
        index_vals = []
        for key in agents: 
            index_vals.append(agents[key].init_lamda_index)
        index_vals = np.array(index_vals)
        
        saveDir = (f'../trainResults/qLearning/deadline_env/arms_{ARMS}_schedule_{SCHEDULE}/episode_{episode+1}')
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        
        np.save(saveDir, index_vals) 
file = open(f'../trainResults/qLearning/deadline_env/arms_{ARMS}_schedule_{SCHEDULE}/trainingInfo.txt', 'w+')
file.write(f'training time: {timer.time() - start:.5f} seconds\n')
file.write(f'training episodes: {EPISODESEND}\n')
file.close()
'''
######################################## QWIC testing #################################################
'''
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

stateArray = []
createStateTable()

for i in range(RUNS):
    envSeeds = np.random.randint(0, 10000, size=ARMS)
    zeroEnvs = {}
    total_reward = []
    ALLEPISODES = np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE) 
    for x in ALLEPISODES:
        EPISODESTRAINED = x
        print(f'doing for episode count : {x}')
        currentTrainedModel = (f'../trainResults/qLearning/deadline_env/arms_{ARMS}_schedule_{SCHEDULE}/episode_{x}.npy')
        qLearningFileName = directory+(f'qLearningResults_arms_{ARMS}_run_{i}_schedule_{SCHEDULE}.csv')

        time = 0
        trained_vals = np.load(currentTrainedModel)
        envs = {}
        rewards = [] 
        index = {}
        agents = {}
        states = {}

        initialize()

        createQLearningAgentsTesting()
        resetEnvs()
        
        while True:
            arm = qLearningTestSelectArms()
            takeActionAndRecordNN(arm) 
            time += 1
            if time == TIMELIMIT: 
                break
            
        total_reward.append((np.cumsum(rewards))[-1])

        print(f'finished Q-learning for trained episodes: {x}')
        print(f'Q-learning for episodes: {x}. rewards: {total_reward[-1]}')

    data = {'episode': np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE), 'cumulative_reward':total_reward}
    df = pd.DataFrame(data=data)
    df.to_csv(qLearningFileName, index=False)
    
    print(f'finished Q-learning deadline scheduling for run {i+1}')
        
'''
###############################- WIBQL training -###############################

def createWibqlStateTable():

    global stateArray
    stateArray.append([0,0])
    for B in range(9+1): 
        for T in range(12+1): 
            state = [T,B]
            if (T != 0):
                if (state !=[12, 0]):
                    stateArray.append(state) 

    stateArray = np.array(stateArray, dtype=np.float32)

def createWibqlAgents():
    global agents, trainingEnvs, stateArray, trainingEnvSeeds

    for key in trainingEnvs:
        agents[key] = WIBQL(numEpisodes=EPISODESEND,episodeLimit=TIMELIMIT,numArms=ARMS,env=trainingEnvs[key],stateTable=stateArray)
    
'''
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

 
episodeRange = np.arange(0, (EPISODESEND + EPISODERANGE), EPISODERANGE) 
trainingEnvs = {}
states = {}
new_states = {}
agents = {}
index = {} 
stateArray = []
trainingEnvSeeds = np.random.randint(0, 10000, size=ARMS)

createWibqlStateTable()
initializeQLearningTraining()
createWibqlAgents()

time = 0
epsilon = 0.1
start = timer.time()

for episode in range(EPISODESEND):
    resetQLearningEnvs()
    print(f"current episode: {episode+1}")

    if (episode == 0):
        index_vals = []
        for key in agents:
            index_vals.append(agents[key].indices)
        index_vals = np.array(index_vals)

        saveDir = (f'../trainResults/wibql/deadline_env/arms_{ARMS}_schedule_{SCHEDULE}/episode_{episode}')
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        np.save(saveDir, index_vals)

    for step in range(TIMELIMIT):
        selectedActions, originalActions, lambda_flag = qLearningChooseArmsTraining()
        TakeActionAndRecordWiqbl(selectedActions)
    
    for key in agents:
        agents[key].updateIndex()

    if (episode+1 in episodeRange) or (episode+1 == EPISODESEND): 
        print(f'saving for episode {episode+1}')
        index_vals = []
        for key in agents:
            index_vals.append(agents[key].indices)
        index_vals = np.array(index_vals)

        saveDir = (f'../trainResults/wibql/deadline_env/arms_{ARMS}_schedule_{SCHEDULE}/episode_{episode+1}')
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        np.save(saveDir, index_vals)

file = open(f'../trainResults/wibql/deadline_env/arms_{ARMS}_schedule_{SCHEDULE}/trainingInfo.txt', 'w+')
file.write(f'training time: {timer.time() - start:.5f} seconds\n')
file.write(f'training episodes: {EPISODESEND}\n')
file.close()        

'''
###############################- WIQBL testing -###############################


def createWiqblAgentsTesting():
    global agents, trained_vals, envs, stateArray, envSeeds
    for key in envs:
        agents[key] = WIBQL(numEpisodes=numEpisodes, episodeLimit=TIMELIMIT, numArms=ARMS, env=envs[key], stateTable=stateArray)
        agents[key].indices = trained_vals[key]

'''
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

stateArray = []
createWibqlStateTable()

for i in range(RUNS):
    envSeeds = np.random.randint(0, 10000, size=ARMS)
    zeroEnvs = {}
    total_reward = []
    ALLEPISODES = np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE) 
    for x in ALLEPISODES:
        EPISODESTRAINED = x
        print(f'doing for episode count : {x}')
        currentTrainedModel = (f'../trainResults/wibql/deadline_env/arms_{ARMS}_schedule_{SCHEDULE}/episode_{x}.npy')
        wibqlFileName = directory+(f'wibqlResults_arms_{ARMS}_run_{i}_schedule_{SCHEDULE}.csv')

        time = 0
        trained_vals = np.load(currentTrainedModel)
        envs = {}
        rewards = [] 
        index = {}
        agents = {}
        states = {}

        initialize()
        createWiqblAgentsTesting()
        resetEnvs()

        while True:
            arm = qLearningTestSelectArms() 
            takeActionAndRecordNN(arm) 
            time += 1
            if time == TIMELIMIT: 
                break
            
        total_reward.append((np.cumsum(rewards))[-1])

        print(f'finished WIBQL for trained episodes: {x}')
        print(f'WIBQL for episodes: {x}. rewards: {total_reward[-1]}')

    data = {'episode': np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE), 'cumulative_reward':total_reward}
    df = pd.DataFrame(data=data)
    df.to_csv(wibqlFileName, index=False)
    print(f'finished WIBQL deadline scheduling for run {i+1}')

'''
##############################- AQL SCHEDULING -######################################
'''

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
            cumReward += reward
        else:
            nextState, reward, done, info = envs[i].step(0)
            stateVals.append(nextState[0])
            stateVals.append(nextState[1])
            cumReward += reward 

    state = stateVals
    state = np.array(state, dtype=np.float32)

    rewards.append((BETA**time)*cumReward)


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