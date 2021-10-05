


import sys
import torch 
import random
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
sys.path.insert(0,'../')
from neurwin import fcnn 
from envs.deadlineSchedulingEnv import deadlineSchedulingEnv
from envs.recoveringBanditsEnv import recoveringBanditsEnv
from envs.sizeAwareIndexEnv import sizeAwareIndexEnv


WIDTH = 12
HEIGHT = 3
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 10

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'


LINEWIDTH = 1
BATCHSIZE = 5
BETA = 0.99
TIMESTEPS = 300
SEED = 42
NORUNS = 50


linestyles=['solid', 'dotted', 'dashed', 'dashdot', (0, (3,1,1,1,1,1))]


fig, axes = plt.subplots(nrows=1,ncols=3, figsize=(WIDTH, HEIGHT), gridspec_kw={'wspace':0.13, 'hspace':0.0}, frameon=False)

#################################### DEADLINE SCHEDULING ####################################

print(f'D_s for deadline')
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROCESSINGCOST = 0.5
EPISODESTRAINED = 2000 

D_vals = np.arange(1,13)
B_vals = np.arange(1,10)

numOfStatesToTest = 5
statesToTest = []

for a in range(numOfStatesToTest):
    state = []
    D = np.random.choice(D_vals)
    B = np.random.choice(B_vals)
    state.append(D)
    state.append(B)
    statesToTest.append(state)


seed = np.random.randint(0, 100000, size = 1000000)
currentActivationCost = np.arange(start=0, stop=6, step=1)

savedDirectory = (f'../trainResults/neurwin/deadline_env/')
modelDir = savedDirectory+(f'seed_{50}_lr_0.001_batchSize_{5}_trainedNumEpisodes_{EPISODESTRAINED}/trained_model.pt')
agent = fcnn(stateSize=2)
agent.load_state_dict(torch.load(modelDir))


D_final = []

for state in statesToTest:
    D_states = []

    for currentAC in currentActivationCost:
        Q_passive_list = []
        Q_active_list = []
        for i in range(NORUNS):

            Q_active = 0
            Q_passive = 0
            env = deadlineSchedulingEnv(seed=seed[i], numEpisodes=1, episodeLimit=TIMESTEPS, maxDeadline=12,
            maxLoad=9, newJobProb=0.7, processingCost=PROCESSINGCOST, train=False, batchSize=5, noiseVar=0)

            env.reset()

            env.arm[0][0] = state[0] 
            env.arm[0][2] = state[0] 
            env.arm[0][1] = state[1]


            nextState, reward, done, info = env.step(1)
            Q_active += BETA**(0)*(reward - currentAC)
     
            for x in range(1, TIMESTEPS):
                index = agent.forward(nextState)
                if index >= currentAC:
                    action = 1
                    nextState, reward, done, info = env.step(action)
                else:
                    action = 0
                    nextState, reward, done, info = env.step(action)

                Q_active += BETA**(x)*(reward - currentAC*action)

            env = deadlineSchedulingEnv(seed=seed[i], numEpisodes=1, episodeLimit=TIMESTEPS, maxDeadline=12,
            maxLoad=9, newJobProb=0.7, processingCost=PROCESSINGCOST, train=False, batchSize=5, noiseVar=0)

            env.reset()
            
            env.arm[0][0] = state[0] 
            env.arm[0][2] = state[0] 
            env.arm[0][1] = state[1]


            nextState, reward, done, info = env.step(0)
            Q_passive += BETA**(0)*(reward)

            for g in range(1, TIMESTEPS):
                index = agent.forward(nextState)
                if index >= currentAC:
                    action = 1
                    nextState, reward, done, info = env.step(action)
                else:
                    action = 0
                    nextState, reward, done, info = env.step(action)

                Q_passive += BETA**(g)*(reward - currentAC*action)

            Q_active_list.append(Q_active)
            Q_passive_list.append(Q_passive)

        average = sum([a_i - b_i for a_i, b_i in zip(Q_active_list, Q_passive_list)]) / NORUNS

        D_states.append(average)
    D_final.append(D_states) 



for q in range(np.shape(statesToTest)[0]):
    axes[0].plot(currentActivationCost, D_final[q], marker='.', label=f's = {statesToTest[q]}', linewidth=LINEWIDTH, linestyle= linestyles[q])

axes[0].set_xticks(currentActivationCost)
axes[0].legend(frameon=False, loc='lower left')


################################### RECOVERING SCHEDULING ###################################
print(f'D_s for recovering')
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


EPISODESTRAINED = 50000
MAXWAIT = 20
selectedActivationFunction = 'A' 
D_final = []
numOfStatesToTest = 5
states = np.random.randint(1,MAXWAIT+1, size=numOfStatesToTest)
seed = np.random.randint(0, 1000, size = 1)[0]
THETA = [10.,0.2,0.0]

currentActivationCost = np.arange(start=0,stop=10.6, step=0.6)

savedDirectory = (f'../trainResults/neurwin/recovering_bandits_env/recovery_function_{selectedActivationFunction}/')
modelDir = savedDirectory+(f'seed_{50}_lr_0.001_batchSize_{5}_trainedNumEpisodes_{EPISODESTRAINED}/trained_model.pt')
agent = fcnn(stateSize=1)
agent.load_state_dict(torch.load(modelDir))


for state in states:
    activation_states = []
    for a in range(np.shape(currentActivationCost)[0]):
        Q_passive = 0
        Q_active = 0
        env = recoveringBanditsEnv(seed=seed, numEpisodes=1, episodeLimit=TIMESTEPS, train=False, 
        batchSize=5, thetaVals=THETA, noiseVar=0.0, maxWait = 20)    

        env.arm[0] = state

        nextState, reward, done, info = env.step(1)
        Q_active += BETA**(0)*(reward - currentActivationCost[a])
        
        for i in range(1,TIMESTEPS):
            index = agent.forward(nextState)

            if index >= currentActivationCost[a]:
                action = 1
                nextState, reward, done, info = env.step(action)
            else:
                action = 0
                nextState, reward, done, info = env.step(action)

            Q_active += BETA**(i)*(reward - currentActivationCost[a]*action)

        env = recoveringBanditsEnv(seed=seed, numEpisodes=1, episodeLimit=TIMESTEPS, train=False, 
        batchSize=5, thetaVals=THETA, noiseVar=0.0, maxWait = 20)    

        env.arm[0] = state

        nextState, reward, done, info = env.step(0)
        Q_passive += BETA**(0)*(reward)
        
        for i in range(1,TIMESTEPS):
            index = agent.forward(nextState)

            if index >= currentActivationCost[a]:
                action = 1
                nextState, reward, done, info = env.step(action)
            else:
                action = 0
                nextState, reward, done, info = env.step(action)

            Q_passive += BETA**(i)*(reward - currentActivationCost[a]*action)

        activation_states.append(Q_active - Q_passive)

    D_final.append(activation_states)


for x in range(len(states)):
    axes[1].plot(currentActivationCost, D_final[x], marker='.', label=f's = {states[x]}.', linewidth=LINEWIDTH, linestyle= linestyles[x])


axes[1].set_xticks(np.arange(0,11))
axes[1].legend(frameon=False, loc='lower left')

################################### SIZE-AWARE SCHEDULING ###################################
print(f'D_s for wireless scheduling')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


MAXLOAD = 1000000
EPISODESTRAINED = 20000
goodStateProb = 0.1
armClass=2

seed = np.random.randint(0, 1000, size = 1000000)


statesToTest = []
numOfStatesToTest = 5
load_vals = np.random.randint(low=300000, high=800000, size=numOfStatesToTest)
for a in range(numOfStatesToTest):
    channelState = np.random.choice([1.0,0.0])
    state = [load_vals[a], channelState]
    statesToTest.append(state)

currentActivationCost = np.arange(start=0, stop=16, step=2)

savedDirectory = (f'../trainResults/neurwin/size_aware_env/case_1/class_{armClass}/')
modelDir = savedDirectory+(f'seed_{50}_lr_0.001_batchSize_{5}_trainedNumEpisodes_{EPISODESTRAINED}/trained_model.pt')
agent = fcnn(stateSize=2)
agent.load_state_dict(torch.load(modelDir))


D_final = []
for state in statesToTest:
    D_states = []

    for cost in currentActivationCost:
        Q_active_list = []
        Q_passive_list = []

        for i in range(NORUNS):

            Q_active = 0
            Q_passive = 0

            env = sizeAwareIndexEnv(numEpisodes=1, HOLDINGCOST=1, seed=seed[i], Training=False,
            r1=8400, r2=33600, q=goodStateProb, case=1, classVal=armClass, load=state[0], noiseVar = 0.0,
            maxLoad = MAXLOAD, batchSize=5, episodeLimit=TIMESTEPS, fixedSizeMDP=False)

            env.reset()
            env.arm[0][0] = state[0]
            env.arm[0][1] = state[1]
            env.channelState[0] = state[1]

            nextState, reward, done, info = env.step(1)
            Q_active += BETA**(0)*(reward - cost)

            x = 1
            while (x < TIMESTEPS and nextState[0] > 0):
            
                index = agent.forward(nextState)
                
                if index >= cost:
                    action = 1
                    nextState, reward, done, info = env.step(action)
                else:
                    action = 0
                    nextState, reward, done, info = env.step(action)

                Q_active += BETA**(x)*(reward - cost*action)

                x += 1

            env = sizeAwareIndexEnv(numEpisodes=1, HOLDINGCOST=1, seed=seed[i], Training=False,
            r1=8400, r2=33600, q=goodStateProb, case=1, classVal=armClass, load=state[0], noiseVar = 0.0,
            maxLoad = MAXLOAD, batchSize=5, episodeLimit=TIMESTEPS, fixedSizeMDP=False)
            
            env.reset()
            env.arm[0][0] = state[0]
            env.arm[0][1] = state[1]
            env.channelState[0] = state[1]

            nextState, reward, done, info = env.step(0)
            Q_passive += BETA**(0)*(reward)

            x = 1
            while (x < TIMESTEPS and nextState[0] > 0):

                index = agent.forward(nextState)

                if index >= cost:
                    action = 1
                    nextState, reward, done, info = env.step(action)
                else:
                    action = 0
                    nextState, reward, done, info = env.step(action)

                Q_passive += BETA**(x)*(reward - cost*action)

                x += 1

            Q_active_list.append(Q_active)
            Q_passive_list.append(Q_passive)

        average = sum([a_i - b_i for a_i, b_i in zip(Q_active_list, Q_passive_list)]) / NORUNS
        D_states.append(average)
    D_final.append(D_states)


for q in range(np.shape(statesToTest)[0]):
    axes[2].plot(currentActivationCost, D_final[q], marker='.', label=f's = {statesToTest[q]}', linewidth=LINEWIDTH, linestyle= linestyles[q])


axes[2].set_xticks(np.arange(0, 16, 2))
axes[2].legend(frameon=False, loc='upper right')


#############################################################################################


axes[0].set_title(f'Deadline Scheduling', weight='bold', fontsize=10)
axes[1].set_title(f'Recovering Bandits', weight='bold', fontsize=10)
axes[2].set_title(f'Wireless Scheduling', weight='bold', fontsize=10)

axes[0].set_ylabel('$D_s(\lambda)$', weight='bold', fontsize=10)
axes[1].set_xlabel('$\lambda$', weight='bold', fontsize=10)
plt.savefig('../plotResults/d_s_results.pdf', bbox_inches='tight')
plt.show()





