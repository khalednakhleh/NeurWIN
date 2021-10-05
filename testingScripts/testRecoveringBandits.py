'''
testing a trained model for scheduling,
and comparing its performance with the 
Recovering Bandits problem.
To test a control policy, uncomment the code corresponding to an algorithm.
'''

import os 
import torch
import random
import itertools 
import numpy as np 
import operator
from scipy.stats import norm 
import scipy.special
import time as timer
import pandas as pd 
import matplotlib.pyplot as plt
import sys
import copy 
sys.path.insert(0,'../')
from neurwin import fcnn 
from envs.recoveringBanditsEnv import recoveringBanditsEnv
from qlearning import qLearningAgent
from wibql import WIBQL
from aql import ProposalQNetwork
import operator
from reinforce import reinforceFcnn

###########################-CONSTANT VALUES-########################################
SEED = 30
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
STATESIZE = 1
numEpisodes = 1
filesSeed = 50 # the seed value in the filename 
################################-PARAMETERS-########################################
SCHEDULE = 1   
TIMELIMIT = 300
NOISEVAR = 0.0
MAXZ = 20
BATCHSIZE = 5
EPISODESEND = 30000
EPISODERANGE = 100
BETA = 0.99
REINFORCELR = 0.001
RUNS = 1
d = 1    # for d-lookahead
CASE = 2 # case 1 is 4 arms, case 2 is 10 arms, case 5 is 100 arms
NOISY = False 
noiseVal = 0.0
if NOISY:
    directory = (f'../testResults/recovering_env/noisy_results/noise_val_{noiseVal}/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    WINNMODELDIR = (f'../trainResults/neurwin/recovering_bandits_env/noise_{noiseVal}_version/')
else:
    directory = (f'../testResults/recovering_env/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    WINNMODELDIR = (f'../trainResults/neurwin/recovering_bandits_env/')

modelDirs = []
THETA = []

classATheta = [10., 0.2, 0.0]
classBTheta = [8.5, 0.4, 0.0]
classCTheta = [7., 0.6, 0.0]
classDTheta = [5.5, 0.8, 0.0]

if CASE == 1:
    ARMS = 4 
    THETA = [classATheta, classBTheta, classCTheta, classDTheta]
    modelDirs.append(WINNMODELDIR+f'recovery_function_A/')
    modelDirs.append(WINNMODELDIR+f'recovery_function_B/')
    modelDirs.append(WINNMODELDIR+f'recovery_function_C/')
    modelDirs.append(WINNMODELDIR+f'recovery_function_D/')

elif CASE == 2: 
    ARMS = 10 
    for i in range(ARMS):
        modelDirs.append(WINNMODELDIR+f'recovery_function_A/')
        THETA.append(classATheta)
        modelDirs.append(WINNMODELDIR+f'recovery_function_B/')
        THETA.append(classBTheta)
        modelDirs.append(WINNMODELDIR+f'recovery_function_C/')
        THETA.append(classCTheta)
        modelDirs.append(WINNMODELDIR+f'recovery_function_D/')
        THETA.append(classDTheta)


elif CASE == 3: 
    ARMS = 12
    for i in range(ARMS):
        modelDirs.append(WINNMODELDIR+f'recovery_function_A/')
        THETA.append(classATheta)
        modelDirs.append(WINNMODELDIR+f'recovery_function_B/')
        THETA.append(classBTheta)
        modelDirs.append(WINNMODELDIR+f'recovery_function_C/')
        THETA.append(classCTheta)
        modelDirs.append(WINNMODELDIR+f'recovery_function_D/')
        THETA.append(classDTheta)

elif CASE == 4: 
    ARMS = 12
    for i in range(ARMS):
        modelDirs.append(WINNMODELDIR+f'recovery_function_A/')
        THETA.append(classATheta)
        modelDirs.append(WINNMODELDIR+f'recovery_function_D/')
        THETA.append(classDTheta)

elif CASE == 5:
    ARMS = 100
    for i in range(ARMS):
        modelDirs.append(WINNMODELDIR+f'recovery_function_A/')
        THETA.append(classATheta)
        modelDirs.append(WINNMODELDIR+f'recovery_function_B/')
        THETA.append(classBTheta)
        modelDirs.append(WINNMODELDIR+f'recovery_function_C/')
        THETA.append(classCTheta)
        modelDirs.append(WINNMODELDIR+f'recovery_function_D/')
        THETA.append(classDTheta)

else:
    print(f'case not list. exiting...')
    exit(1)


REINFORCEDIR = (f'../trainResults/reinforce/recovering_env/arms_{ARMS}_schedule_{SCHEDULE}/')
WOLPMODELDIR = (f'../trainResults/wolp_ddpg/recovering_bandits_env/arms_{ARMS}_schedule_{SCHEDULE}/')
AQLDIR = (f'../trainResults/aql/recovering_env/arms_{ARMS}_schedule_{SCHEDULE}/')

readMeFileName = (f'{directory}'+'readme.txt')
readMeFile = open(readMeFileName, 'a')
readMeFile.write(f'\nNumber of arms: {ARMS}\n')
readMeFile.close()


def initialize():
    global envSeeds, envs
    for i in range(ARMS):
        env = recoveringBanditsEnv(seed=envSeeds[i], numEpisodes=numEpisodes, episodeLimit=TIMELIMIT, train=False, 
batchSize=EPISODESEND, thetaVals=THETA[i], noiseVar=NOISEVAR, maxWait = MAXZ)

        envs[i] = env

def initializeNN():
    global envsSeeds, envs, agents

    for i in range(ARMS):
        env = recoveringBanditsEnv(seed=envSeeds[i], numEpisodes=numEpisodes, episodeLimit=TIMELIMIT, train=False, 
batchSize=EPISODESEND, thetaVals=THETA[i], noiseVar=NOISEVAR, maxWait = MAXZ)           
        agent = fcnn(stateSize=1)
        agent.load_state_dict(torch.load(modelDirs[i]+currentTrainedModel))
        agent.eval()
        envs[i] = env
        agents[i] = agent 

def takeActionAndRecordNN(arms):
    global rewards, time, states, envs
    
    finalReward = 0
    for arm in arms:
        nextState, reward, done, info = envs[arm].step(1)
        finalReward += reward
        states[arm] = nextState
    
    for key in envs:
        if key in arms:
            pass
        else:
            nextState, redundantReward, done, info = envs[key].step(0)
            finalReward += redundantReward
            states[key] = nextState

    rewards.append((BETA**time)*finalReward)


def getSelection(index):
 
    result = []
    copyIndex = index.copy()

    for i in range(SCHEDULE):
        result.append(max(copyIndex.items(), key=operator.itemgetter(1))[0])
        del copyIndex[result[i]]

    choice = result
    return choice 

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

def activateArmsAndRecordReward(arms):
    global rewards, t, time, states, envs 
    
    finalReward = 0
    for arm in arms:
        nextState, reward, done, info = envs[arm].step(1)
        finalReward += reward
        states[arm] = nextState

    for key in envs:
        if key in arms:
            pass
        else:
            nextState, redundantReward, done, info = envs[key].step(0)
            states[key] = nextState
    rewards.append((BETA**t)*finalReward)

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
        val = envs[key].reset()[0]
        state.append(val)
    state = np.array(state, dtype=np.float32)

def REINFORCETakeActionAndRecordReward():
    global rewards, state, reinforceAgent, envs, actionTable

    cumReward = 0
    stateVals = []

    action_probs = reinforceAgent.forward(state).detach().numpy()
    action = np.random.choice(np.arange(len(actionTable)), p=action_probs)
    actionVector = actionTable[action]

    for i in range(len(actionVector)):
        if actionVector[i] == 1:
            nextState, reward, done, info = envs[i].step(1)
            stateVals.append(nextState[0])
            cumReward += reward
        else:
            nextState, redundantReward, done, info = envs[i].step(0)
            cumReward += redundantReward 
            stateVals.append(nextState[0])

    state = stateVals
    state = np.array(state, dtype=np.float32)

    rewards.append((BETA**time)*cumReward)

def initializeREINFORCE(): 
    global envSeeds, envs

    for i in range(ARMS):
        env = recoveringBanditsEnv(seed=envSeeds[i], numEpisodes=numEpisodes, episodeLimit=TIMELIMIT, train=False, 
batchSize=EPISODESEND, thetaVals=THETA[i], noiseVar=NOISEVAR, maxWait = MAXZ)        
        envs[i] = env

def ZSeq(myarm, myseq, Z):
    d = len(myseq)
    z = np.zeros(d+1)
    z[0] = myarm.arm[0]
    
    for i in range(1,d+1):
        if myseq[i-1]==myarm:
            z[i] = 1
        else:
            z[i] = min(z[i-1] +1, Z)
    return z

def ZPlayedSeq(myseq, Z):
    zseqs = {}
    for arm in myseq:
        zseqs[arm] = ZSeq(arm, myseq, Z)
    allzs = [zseqs[myseq[i]][i] for i in range(len(myseq))]
    return allzs


def updateStateVals():
    global envs 
    for arm in envs: 
        arm.arm[0] = min(arm.arm[0] + 1, MAXZ)


def takeActionAndRecordTS(arms):
    global rewards, time, states, envs
    
    finalReward = 0
    for arm in arms:
        nextState, reward, done, info = envs[arm].step(1)
        finalReward += reward
        states[arm] = nextState
        envs[arm].UpdatePosterior(reward)
    
    for key in range(len(envs)):
       
        if key in arms:
            pass
        else:
            nextState, redundantReward, done, info = envs[key].step(0)
            finalReward += redundantReward
            states[key] = nextState
    
    rewards.append((BETA**time)*finalReward)


###########################-    TESTING SETTINGS    -######################################
###########################################################################################

def chooseBestSequence(sequences):
    global envs

    sequenceReward = []
    originalZ = [envs[i].arm[0] for i in range(len(envs))]

    for i in sequences:
        rewards = []
        timer = 0
        for x in i:
            nextState, reward, done, info = envs[x].step(1)

            for key in envs:
                if key == x:
                    pass
                else:
                    nextState, redundantReward, done, info = envs[key].step(0)

            rewards.append((BETA**timer)*reward)
            timer += 1

        for key in envs:
            envs[key].arm[0] = originalZ[key]

        sequenceReward.append(sum(rewards))

    result = sequenceReward.index(max(sequenceReward))

    return result 

def dLookAheadPolicy():
    global envs, d, T, maxWait, rewards 
    keyList = list(envs.keys())
    allSequences = list(itertools.product(keyList, repeat=d))

    bestSequenceIndex = chooseBestSequence(allSequences)

    return allSequences[bestSequenceIndex]

def takeActionDLookAhead(sequencePoint):
    global selectedSequence, rewards, envs, t

    armToActivate = selectedSequence[sequencePoint]

    nextState, activationReward, done, info = envs[armToActivate].step(1)
    states[armToActivate] = nextState

    for key in envs:
        if key == armToActivate:
            pass
        else:
            nextState, reward, done, info = envs[key].step(0)
            activationReward += reward 
            states[key] = nextState
    
    rewards.append((BETA**t)*activationReward)


####################### 1-lookahead scheduling policy ################################

def lookAhead1SelectArms(): 
    global envs, states
    rewards = {}
    stateVals = [envs[i].arm[0] for i in range(len(envs))]

    for q in envs:
        rewards[q] = envs[q]._calReward(1, envs[q].arm[0])

    choice = getSelection(rewards)

    return choice

def takeAction1LookAhead(arms):
    global rewards, envs, t
    
    reward = 0

    for arm in arms:
        nextState, activationReward, done, info = envs[arm].step(1)
        reward += activationReward

    for key in envs:
        if key in arms:
            pass
        else:
            nextState, redundantReward, done, info = envs[key].step(0)
            reward += redundantReward

    rewards.append((BETA**t)*reward) 
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
    states = {}

    print(f'doing the 1-lookahead rewards\' value for run {i+1}')
    initialize()
    resetEnvs()
    selectedSequence = []
    
    for t in range(0, TIMELIMIT):

        arms = lookAhead1SelectArms()
        takeAction1LookAhead(arms)

    total_reward = (np.cumsum(rewards))[-1]
    print(f'total reward: {total_reward}')
    cumReward.append(total_reward)

data = {'run': range(RUNS), 'cumulative_reward':cumReward}

df = pd.DataFrame(data=data)
dLookAheadFileName = (f'{directory}'+f'dLookAheadResults_arms_{ARMS}_timeLimit_{TIMELIMIT}_d_{d}_case_{CASE}_schedule_{SCHEDULE}.csv')
df.to_csv(dLookAheadFileName, index=False)
'''
####################### d-lookahead scheduling policy ################################
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
    states = {}
    print(f'doing the d-lookahead rewards\' value for run {i+1} for d={d}')
    initialize()
    resetEnvs()
    selectedSequence = []
    
    for t in range(0, TIMELIMIT):
        sequencePoint = t % d 
        if sequencePoint == 0:
            selectedSequence = list(dLookAheadPolicy())

        takeActionDLookAhead(sequencePoint)

    total_reward = (np.cumsum(rewards))[-1]
    print(f'total reward: {total_reward}')
    cumReward.append(total_reward)
data = {'run': range(RUNS), 'cumulative_reward':cumReward}
df = pd.DataFrame(data=data)
dLookAheadFileName = (f'{directory}'+f'dLookAheadResults_arms_{ARMS}_timeLimit_{TIMELIMIT}_d_{d}_case_{CASE}_schedule_{SCHEDULE}.csv')
df.to_csv(dLookAheadFileName, index=False)
'''
###########################################################################################
##############################- NeurWIN Scheduling -#######################################
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
        print(f'doing for episode count : {x}')
        currentTrainedModel = (f'seed_{filesSeed}_lr_0.001_batchSize_{BATCHSIZE}_trainedNumEpisodes_{EPISODESTRAINED}/trained_model.pt')
        nnFileName = directory+(f'nnIndexResults_arms_{ARMS}_batchSize_{BATCHSIZE}_case_{CASE}_timeLimit_{TIMELIMIT}_schedule_{SCHEDULE}.csv')

        time = 0
        envs = {}
        rewards = []
        index = {}
        agents = {}
        states = {}
        indexNN = {}
        initializeNN()
        resetEnvs()
        
        while True:
            arm = calculateIndexNeuralNetwork()
            takeActionAndRecordNN(arm)
            time += 1
            if time == TIMELIMIT: 
                break
            
        total_reward.append((np.cumsum(rewards))[-1])

        print(f'finished NN for trained episodes: {x}')
        print(f'NeurWIN for episodes: {x}. rewards: {total_reward[-1]}')

    data = {'episode': np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE), 'cumulative_reward':total_reward}
    df = pd.DataFrame(data=data)
    df.to_csv(nnFileName, index=False)
    print(f'finished NN recovering bandits scheduling for run {i+1}') #case {CASE} for number of episodes: {EPISODESTRAINED}')
'''
############################ REINFORCE Scheduling #####################################
'''
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

actionTable = getActionTableLength()

for i in range(RUNS):
    envSeeds = np.random.randint(0, 10000, size=ARMS)
    ALLEPISODES = np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE) 
    zeroEnvs = {}
    total_reward = []
    for x in ALLEPISODES:

        EPISODESTRAINED = x
        REINFORCEMODELDIR = REINFORCEDIR+(f'seed_{filesSeed}_lr_{REINFORCELR}_batchSize_{BATCHSIZE}_trainedNumEpisodes_{EPISODESTRAINED}/trained_model.pt')
        reinforceFileName = directory+(f'reinforceResults_arms_{ARMS}_batchSize_{BATCHSIZE}_lr_{REINFORCELR}_run_{i}_schedule_{SCHEDULE}.csv')
        time = 0
        
        envs = {}
        rewards = []
        state = []
        initializeREINFORCE()

        resetREINFORCEEnvs()
        reinforceAgent = reinforceFcnn(stateDim=ARMS, actionDim=ARMS, hidden1=96, hidden2=48) 
        reinforceAgent.load_state_dict(torch.load(REINFORCEMODELDIR))
        reinforceAgent.eval()
        
        while True:
            REINFORCETakeActionAndRecordReward()
            time += 1
            if time == TIMELIMIT: # time limit reached
                break

        total_reward.append((np.cumsum(rewards))[-1])
        print(f'finished for trained episodes: {x}. cumulative_reward: {total_reward[-1]}')

    data = {'episode': np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE), 'cumulative_reward':total_reward}
    df = pd.DataFrame(data=data)
    df.to_csv(reinforceFileName, index=False)
    print(f'finished REINFOCE scheduling for run {i+1}') #case {CASE} for number of episodes: {EPISODESTRAINED}')

'''
############################### Q-learning training ########################################


def initializeQLearningTraining():
    global trainingEnvSeeds, trainingEnvs
    for i in range(ARMS):
        env = recoveringBanditsEnv(seed=trainingEnvSeeds[i], numEpisodes=EPISODESEND, episodeLimit=TIMELIMIT, train=True, 
batchSize=BATCHSIZE, thetaVals=THETA[i], noiseVar=NOISEVAR, maxWait = MAXZ)

        trainingEnvs[i] = env

def createAgents():
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

def resetQLearningEnvs():
    global states, trainingEnvs
    for key in trainingEnvs:
        state = trainingEnvs[key].reset()
        states[key] = state


def createStateTable():
    global stateArray 
    for x in range(MAXZ):
            state = x+1
            stateArray.append(state)  

    stateArray = np.array(stateArray, dtype=np.uint32)

'''
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

stateArray = []

episodeRange = np.arange(0, (EPISODESEND + EPISODERANGE), EPISODERANGE) 
trainingEnvs = {}
states = {}
new_states = {}
agents = {}
index = {} 
trainingEnvSeeds = np.random.randint(0, 10000, size=ARMS)
createStateTable()
initializeQLearningTraining()
createAgents()
time = 0
epsilon = 1 
start = timer.time()
for episode in range(EPISODESEND):
    resetQLearningEnvs()
    print(f'current episode: {episode+1}')

    if (episode == 0):
        index_vals = []
        lamda_qTable = []
        for key in agents: 
            index_vals.append(agents[key].init_lamda_index)
            lamda_qTable.append(agents[key].lamda_qTable)
        index_vals = np.array(index_vals)
        lamda_qTable = np.array(lamda_qTable)
        
        saveDir = (f'../trainResults/qLearning/recovering_bandits_env/arms_{ARMS}_schedule_{SCHEDULE}/episode_{episode}')
        saveDirQTable = (f'../trainResults/qLearning/recovering_bandits_env/arms_{ARMS}_schedule_{SCHEDULE}/qTable_{episode}')
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
            os.makedirs(saveDirQTable)
        
        np.save(saveDir, index_vals) 
        np.save(saveDirQTable, lamda_qTable)

    for step in range(TIMELIMIT):

        selectedActions, originalActions, lambda_flag = qLearningChooseArmsTraining()
        TakeActionAndRecordQLearning(selectedActions, originalActions, lambda_flag)
        time += 1
        epsilon = min(1,2*(time)**(-0.5))

    if (episode+1 in episodeRange) or (episode+1 == EPISODESEND): 
        print(f'saving for episode {episode+1}')
        index_vals = []
        lamda_qTable = []
        for key in agents:
            index_vals.append(agents[key].init_lamda_index)
            lamda_qTable.append(agents[key].lamda_qTable)
        index_vals = np.array(index_vals)
        lamda_qTable = np.array(lamda_qTable)

        saveDir = (f'../trainResults/qLearning/recovering_bandits_env/arms_{ARMS}_schedule_{SCHEDULE}/episode_{episode+1}')
        saveDirQTable = (f'../trainResults/qLearning/recovering_bandits_env/arms_{ARMS}_schedule_{SCHEDULE}/qTable_{episode+1}')
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        np.save(saveDir, index_vals)  # saving current lambda mapping table
        np.save(saveDirQTable, lamda_qTable)

file = open(f'../trainResults/qLearning/recovering_bandits_env/arms_{ARMS}_schedule_{SCHEDULE}/trainingInfo.txt', 'w+')
file.write(f'training time: {timer.time() - start:.5f} seconds\n')
file.write(f'training episodes: {EPISODESEND}\n')
file.close()
'''
######################################## Q-learning testing #################################################

def createAgentsTesting():
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
        currentTrainedModel = (f'../trainResults/qLearning/recovering_bandits_env/arms_{ARMS}_schedule_{SCHEDULE}/episode_{x}.npy')
        qLearningFileName = directory+(f'qLearningResults_arms_{ARMS}_run_{i}_schedule_{SCHEDULE}.csv')

        time = 0
        trained_vals = np.load(currentTrainedModel)
        envs = {}
        rewards = [] 
        index = {}
        agents = {}
        states = {}

        initialize()
        createAgentsTesting()
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
    print(f'finished Q-learning recovering bandits scheduling for run {i+1}') #case {CASE} for number of episodes: {EPISODESTRAINED}')


'''
########################### WIBQL training ###########################
''' 
def createWibqlAgents():
    global agents, trainingEnvs, stateArray, trainingEnvSeeds

    for key in trainingEnvs:
        agents[key] = WIBQL(numEpisodes=EPISODESEND,episodeLimit=TIMELIMIT,numArms=ARMS,env=trainingEnvs[key],stateTable=stateArray)

def TakeActionAndRecordWiqbl(selectedActions):
    global rewards, time, trainingEnvs

    cumReward = 0
    for i in trainingEnvs:
        nextState = agents[i]._takeAction(selectedActions[i])
        states[i] = nextState


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

stateArray = []

episodeRange = np.arange(0, (EPISODESEND + EPISODERANGE), EPISODERANGE) 
trainingEnvs = {}
states = {}
new_states = {}
agents = {}
index = {} 
trainingEnvSeeds = np.random.randint(0, 10000, size=ARMS)
createStateTable()
initializeQLearningTraining()
createWibqlAgents()
time = 0
epsilon = 0.1

start = timer.time()
for episode in range(EPISODESEND):
    resetQLearningEnvs()
    print(f'current episode: {episode+1}')

    if (episode == 0):
        index_vals = []
        qTable = []
        for key in agents: 
            index_vals.append(agents[key].indices)
            qTable.append(agents[key].qTable)
        index_vals = np.array(index_vals)
        qTable = np.array(qTable)
        
        saveDir = (f'../trainResults/wibql/recovering_bandits_env/arms_{ARMS}_schedule_{SCHEDULE}/episode_{episode}')
        saveDirQTable = (f'../trainResults/wibql/recovering_bandits_env/arms_{ARMS}_schedule_{SCHEDULE}/qTable_{episode}')
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
            os.makedirs(saveDirQTable)
        
        np.save(saveDir, index_vals) 
        np.save(saveDirQTable, qTable)

    for step in range(TIMELIMIT):

        selectedActions, originalActions, lambda_flag = qLearningChooseArmsTraining()
        TakeActionAndRecordWiqbl(selectedActions)
        time += 1

    for key in agents: 
        agents[key].updateIndex()

    if (episode+1 in episodeRange) or (episode+1 == EPISODESEND):
        print(f'saving for episode {episode+1}')
        index_vals = []
        qTable = []
        for key in agents:
            index_vals.append(agents[key].indices)
            qTable.append(agents[key].qTable)
        index_vals = np.array(index_vals)
        qTable = np.array(qTable)

        saveDir = (f'../trainResults/wibql/recovering_bandits_env/arms_{ARMS}_schedule_{SCHEDULE}/episode_{episode+1}')
        saveDirQTable = (f'../trainResults/wibql/recovering_bandits_env/arms_{ARMS}_schedule_{SCHEDULE}/qTable_{episode+1}')
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        np.save(saveDir, index_vals)
        np.save(saveDirQTable, qTable)

file = open(f'../trainResults/wibql/recovering_bandits_env/arms_{ARMS}_schedule_{SCHEDULE}/trainingInfo.txt', 'w+')
file.write(f'training time: {timer.time() - start:.5f} seconds\n')
file.write(f'training episodes: {EPISODESEND}\n')
file.close()

'''
########################### WIBQL testing ###########################

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
createStateTable()

for i in range(RUNS):
    envSeeds = np.random.randint(0, 10000, size=ARMS)
    zeroEnvs = {}
    total_reward = []
    ALLEPISODES = np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE) 
    for x in ALLEPISODES:
        EPISODESTRAINED = x
        print(f'doing for episode count : {x}')
        currentTrainedModel = (f'../trainResults/wibql/recovering_bandits_env/arms_{ARMS}_schedule_{SCHEDULE}/episode_{x}.npy')
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
    print(f'finished WIBQL recovering bandits scheduling for run {i+1}') #case {CASE} for number of episodes: {EPISODESTRAINED}')

'''

########################## AQL TESTING ###############################


def aqlTakeActionAndRecordReward():
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
            cumReward += reward
        else:
            nextState, redundantReward, done, info = envs[i].step(0)
            cumReward += redundantReward 
            stateVals.append(nextState[0])

    state = stateVals
    state = np.array(state, dtype=np.float32)

    rewards.append((BETA**time)*cumReward)

'''
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#actionTable = getActionTableLength()

# 64,32 for 4 choose 1, 92,48 for 10 choose 1, 512,150 100 choose 25
DIM1 = 92
DIM2 = 48

for i in range(RUNS):
    envSeeds = np.random.randint(0, 10000, size=ARMS)
    ALLEPISODES = np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE) 
    total_reward = []
    for x in ALLEPISODES:

        EPISODESTRAINED = x
        AQLMODELDIR = AQLDIR+(f'seed_{filesSeed}_lr_{REINFORCELR}_trainedNumEpisodes_{EPISODESTRAINED}/trained_model.pt')
        AQLFileName = directory+(f'aqlResults_arms_{ARMS}_run_{i}_schedule_{SCHEDULE}.csv')
        time = 0
        
        envs = {}
        rewards = []
        state = []
        initializeREINFORCE()

        resetREINFORCEEnvs()
        aqlAgent = ProposalQNetwork(ARMS, ARMS, DIM1, DIM2)
        aqlAgent.load_state_dict(torch.load(AQLMODELDIR))
        aqlAgent.eval()
        
        while True:
            aqlTakeActionAndRecordReward()
            time += 1
            if time == TIMELIMIT: 
                break

        total_reward.append((np.cumsum(rewards))[-1])
        print(f'finished for trained episodes: {x}. cumulative_reward: {total_reward[-1]}')

    data = {'episode': np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE), 'cumulative_reward':total_reward}
    df = pd.DataFrame(data=data)
    df.to_csv(AQLFileName, index=False)
    print(f'finished AQL scheduling for run {i+1}') #case {CASE} for number of episodes: {EPISODESTRAINED}')


'''
############################### UCB ###############################

def takeActionAndRecordUCB(arms):
    global states, envs
    
    finalReward = 0

    for arm in arms:
        
        nextState, reward, done, info = envs[arm].step(1)
        
        finalReward += reward
        states[arm] = nextState
        envs[arm].UpdatePosterior(reward)
    
    for key in envs:
        if key in arms:
            pass
        else:
            nextState, redundantReward, done, info = envs[key].step(0)
            finalReward += redundantReward
            states[key] = nextState


def takeActionAndRecordUCBTesting(arms):
    global states, envs
    
    finalReward = 0
    for arm in arms:
        nextState, reward, done, info = envs[arm].step(1)
        envs[arm].episodeTime -= 1
        
        finalReward += reward
        states[arm] = nextState

    for key in envs:
        if key in arms:
            pass
        else:
            nextState, redundantReward, done, info = envs[key].step(0)
            finalReward += redundantReward
            states[key] = nextState
            envs[arm].episodeTime -= 1

    return finalReward

def CalcUCB(time, K, Z):
    global envs

    for myarm in envs:
        m = envs[myarm].model
        pred = m.predict(np.array([[envs[myarm].arm[0]]]))
        
        cb = np.sqrt(2*np.log(time**2*Z*K))
        ucb = pred[0] + cb*np.sqrt(pred[1])
        envs[myarm].ucb = ucb


def getUCB():
    global ARMS, envs
    ucbs = {}
    for i in range(ARMS):
        ucbs[i] = envs[i].ucb

    return ucbs 

def getUCBTest():
    global ARMS
    ucbsTest = {}
    for i in range(ARMS):
        ucbsTest[i] = envs[i].ucb

    return ucbsTest

def trainUCB(currentEpisode):
    global trainingTime, SCHEDULE, envs
    tempTime = 0 
    if currentEpisode == 0:
        if SCHEDULE == 1: 
            for tempOne in range(0, ARMS):
                currentArm = tempOne
                takeActionAndRecordUCB([currentArm])
                tempTime += 1
                trainingTime += 1 
        else: 
            for t in range(0, 4): 
                currentArms = ArmsToBegin[25*t:25*(t+1)]
                takeActionAndRecordUCB(currentArms)
                tempTime += 1
                trainingTime += 1 

        for t in range(tempTime, TIMELIMIT):

            CalcUCB(time=trainingTime, Z=MAXZ, K=ARMS)
            ucbs = getUCB()

            myarms = getSelection(ucbs)
            takeActionAndRecordUCB(myarms)

            trainingTime += 1
    else:            
        for t in range(0, TIMELIMIT):
            
            CalcUCB(time=trainingTime, Z=MAXZ, K=ARMS)
            ucbs = getUCB()

            myarms = getSelection(ucbs)
            takeActionAndRecordUCB(myarms)

            trainingTime += 1

def calRewardUCBTesting():
    global envs, rewards

    for key in envs:
        envs[key].arm[0] = 1 

    for testTime in range(0, TIMELIMIT):
        
        CalcUCB(time=trainingTime, Z=MAXZ, K=ARMS)
        ucbsTest = getUCBTest() 
        arms = getSelection(ucbsTest)
        rewardVal = takeActionAndRecordUCBTesting(arms)

        rewards.append((BETA**testTime)*rewardVal)


def initializeUCBTSTraining():
    global trainingEnvSeeds, envs
    for i in range(ARMS):
        env = recoveringBanditsEnv(seed=trainingEnvSeeds[i], numEpisodes=EPISODESEND, episodeLimit=TIMELIMIT, train=True, 
batchSize=BATCHSIZE, thetaVals=THETA[i], noiseVar=NOISEVAR, maxWait = MAXZ)

        envs[i] = env


def nonTrainedUCBTS():
    rewards = []
    trainingSeeds = np.random.randint(0, 10000, size=ARMS)
    startingEnvs = {}
    for i in range(ARMS):
        startingEnv = recoveringBanditsEnv(seed=trainingSeeds[i], numEpisodes=EPISODESEND, episodeLimit=TIMELIMIT, train=True, 
batchSize=BATCHSIZE, thetaVals=THETA[i], noiseVar=NOISEVAR, maxWait = MAXZ)

        startingEnvs[i] = startingEnv
    ucbtslist = dict.fromkeys(np.arange(0,ARMS), 0)
    for t in range(0, TIMELIMIT):
 
        selectedArms = getSelection(ucbtslist)

        finalReward = 0
        for arm in selectedArms:
            nextState, reward, done, info = startingEnvs[arm].step(1)
            
            finalReward += reward

        for key in startingEnvs:
            if key in selectedArms:
                pass
            else:
                nextState, redundantReward, done, info = startingEnvs[key].step(0)
                finalReward += redundantReward

        rewards.append((BETA**t)*finalReward)
    
    return rewards 

'''
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


cumReward = []
ArmsToBegin = np.arange(0, ARMS)
episodeRange = np.arange(0, (EPISODESEND + EPISODERANGE), EPISODERANGE) 
envs = {}

trainingEnvSeeds = np.random.randint(0, 10000, size=ARMS)

states = {}

initializeUCBTSTraining() 

trainingTime = 0

start = timer.time()

rewards = nonTrainedUCBTS()  
total_reward = (np.cumsum(rewards))[-1]
cumReward.append(total_reward)
print(f'non-trained result before running UCB. rewards: {cumReward[-1]}')


for episode in range(EPISODESEND):
    resetEnvs()
    
    print(f'current episode: {episode+1}')
    trainUCB(currentEpisode=episode) # training loop

    if (episode+1 in episodeRange) or (episode+1 == EPISODESEND):
        rewards = []
        print(f'Testing UCB for episode: {episode+1}')
        calRewardUCBTesting() 
        total_reward = (np.cumsum(rewards))[-1]
        cumReward.append(total_reward)
        print(f'Finished UCB for episodes: {episode+1}. rewards: {cumReward[-1]}')

end = timer.time()

data = {'episode': np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE), 'cumulative_reward':cumReward}
df = pd.DataFrame(data=data)
UCBFileName = (f'{directory}'+f'ucbResults_arms_{ARMS}_timeLimit_{TIMELIMIT}_d_{d}_case_{CASE}_schedule_{SCHEDULE}.csv')
df.to_csv(UCBFileName, index=False)
print(f'finished UCB recovering bandits scheduling') #case {CASE} for number of episodes: {EPISODESTRAINED}')
print(f'\ntraining time: {trainingTime}')

'''

############################ Thomspon sampling ######################
def takeActionAndRecordTS(arms):
    global states, envs
    
    finalReward = 0
    for arm in arms:
        nextState, reward, done, info = envs[arm].step(1)
        finalReward += reward
        states[arm] = nextState
        envs[arm].UpdatePosterior(reward)
    
    for key in envs:
        if key in arms:
            pass
        else:
            nextState, redundantReward, done, info = envs[key].step(0)
            finalReward += redundantReward
            states[key] = nextState


def calcTS(t, K, Z):
    global envs

    for myarm in envs:
        m = envs[myarm].model
        pred = m.predict(np.array([[envs[myarm].arm[0]]]))
        mean = pred[0]
        sd = np.sqrt(pred[1])
        mysample = norm.rvs(mean, sd)
        envs[myarm].ts = mysample


def getTS():
    global envs, ARMS 
    tsList = {}
    for i in range(ARMS):
        tsList[i] = envs[i].ts

    return tsList


def trainTS(currentEpisode):
    global trainingTime, SCHEDULE, envs
    tempTime = 0 
    if currentEpisode == 0:
        if SCHEDULE == 1:
            for tempOne in range(0, ARMS):
                currentArm = tempOne
                takeActionAndRecordTS([currentArm])
                tempTime += 1
                trainingTime += 1 
        else: 
            for t in range(0, 4): 
                currentArms = ArmsToBegin[25*t:25*(t+1)]
                takeActionAndRecordTS(currentArms)
                tempTime += 1
                trainingTime += 1 

        for t in range(tempTime, TIMELIMIT):
            
            calcTS(t=trainingTime, Z=MAXZ, K=ARMS)
            tsList = getTS()
            
            myarms = getSelection(tsList)
            takeActionAndRecordTS(myarms)

            trainingTime += 1 
    else:
        for t in range(0, TIMELIMIT):
            
            calcTS(t=trainingTime, Z=MAXZ, K=ARMS)
            tsList = getTS()
            
            myarms = getSelection(tsList)
            takeActionAndRecordTS(myarms)

            trainingTime += 1

def getTSTest():
    global ARMS
    tsTestList = {}
    for i in range(ARMS):
        tsTestList[i] = envs[i].ts
    
    return tsTestList

def takeActionAndRecordTSTesting(arms):
    global states, envs
    
    finalReward = 0
    for arm in arms:
        nextState, reward, done, info = envs[arm].step(1)
        envs[arm].episodeTime -= 1
        finalReward += reward
        states[arm] = nextState

    for key in envs:
        if key in arms:
            pass
        else:
            nextState, redundantReward, done, info = envs[key].step(0)
            finalReward += redundantReward
            states[key] = nextState
            envs[arm].episodeTime -= 1

    return finalReward

def calRewardTSTesting():
    global envs, rewards
    
    for key in envs:
        envs[key].arm[0] = 1 

    for testTime in range(0, TIMELIMIT):
        calcTS(t=trainingTime, Z=MAXZ, K=ARMS)
        tsTestList = getTSTest()
        arms = getSelection(tsTestList)
        rewardVal = takeActionAndRecordTSTesting(arms)
        rewards.append((BETA**testTime)*rewardVal)

'''
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


cumReward = []
ArmsToBegin = np.arange(0, ARMS)
episodeRange = np.arange(0, (EPISODESEND + EPISODERANGE), EPISODERANGE) 
envs = {}

trainingEnvSeeds = np.random.randint(0, 10000, size=ARMS)

states = {}

initializeUCBTSTraining() 

trainingTime = 0

start = timer.time()

rewards = nonTrainedUCBTS() 
total_reward = (np.cumsum(rewards))[-1]
cumReward.append(total_reward)
print(f'non-trained result before running TS. rewards: {cumReward[-1]}')


for episode in range(EPISODESEND):
    resetEnvs()
    print(f'current episode: {episode+1}')
    trainTS(currentEpisode=episode) 

    if (episode+1 in episodeRange) or (episode+1 == EPISODESEND):
        rewards = [] 
        print(f'Testing TS for episode: {episode+1}')
        calRewardTSTesting() # test TS for specified episode interval
        total_reward = (np.cumsum(rewards))[-1]
        cumReward.append(total_reward)
        print(f'Finished TS for episodes: {episode+1}. rewards: {cumReward[-1]}')

end = timer.time()

data = {'episode': np.arange(0, EPISODESEND+EPISODERANGE, EPISODERANGE), 'cumulative_reward':cumReward}
df = pd.DataFrame(data=data)
TSFileName = (f'{directory}'+f'tsResults_arms_{ARMS}_timeLimit_{TIMELIMIT}_d_{d}_case_{CASE}_schedule_{SCHEDULE}.csv')
df.to_csv(TSFileName, index=False)
print(f'finished TS recovering bandits scheduling') #case {CASE} for number of episodes: {EPISODESTRAINED}')
print(f'\ntraining time: {trainingTime}')

'''
