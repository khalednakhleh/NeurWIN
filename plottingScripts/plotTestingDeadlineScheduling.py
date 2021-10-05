
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


WIDTH = 12
HEIGHT = 3
plt.rcParams['font.size'] = 14
plt.rcParams['legend.fontsize'] = 14

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'

SCHEDULELIST = [1,1,25]    
ARMSLIST = [4, 10, 100]        
INTERVALLIST = [10, 10, 20]


########################## CONSTANTS ######################
LINEWIDTH = 1.75
BATCHSIZE = 5
BETA = 0.99
EPISODEEND = 2000
RUNS = 50
TIMELIMIT = 300
REINFORCELR = 0.001


def plotDeadlineIndex(ARMS, SCHEDULE):

    deadlineFileName = (f'../testResults/deadline_env/deadlineIndexResults_arms_{ARMS}_timeLimit_{TIMELIMIT}_schedule_{SCHEDULE}.csv')
    df = pd.read_csv(deadlineFileName)
    deadlineRewards = df.iloc[:,1]
    deadline5Percentile, deadline95Percentile = np.percentile(deadlineRewards[0:RUNS], [5, 95])

    deadlineRewards = sum(deadlineRewards[0:RUNS] / RUNS)
    
    return deadlineRewards, deadline5Percentile, deadline95Percentile

def plotNeurWIN(ARMS, SCHEDULE, INTERVAL):

    nnRewards = []

    for i in range(RUNS):
        NeurWINFileName = (f'../testResults/deadline_env/nnIndexResults_arms_{ARMS}_batchSize_{BATCHSIZE}_run_{i}_timeLimit_{TIMELIMIT}_schedule_{SCHEDULE}.csv')
        df = pd.read_csv(NeurWINFileName)
        runReward = df.iloc[:, 1]
        nnRewards.append(runReward)

    nnVal = np.sum(nnRewards, 0) / RUNS

    nnRewards = np.transpose(nnRewards)

    percentile5 = np.percentile(nnRewards, 5, axis=1)
    percentile95 = np.percentile(nnRewards, 95, axis=1)

    return nnVal, percentile5, percentile95

def plotWolp(ARMS, SCHEDULE, INTERVAL):

    wolpRewards = []

    for i in range(RUNS):
        if ARMS == 100:
            wolpFileName = (f'../testResults/deadline_env/wolpResults_arms_{ARMS}_batchSize_{BATCHSIZE}_run_{i}_timeLimit_{TIMELIMIT}_schedule_{SCHEDULE}.csv')
        else:
            wolpFileName = (f'../wolpertinger_ddpg/testResults/deadline_ddpg_results_{ARMS}_arms_choose_{SCHEDULE}_run_{i}.csv')
        df = pd.read_csv(wolpFileName)
        runReward = df.iloc[:, 1]
        wolpRewards.append(runReward)

    wolpVal = np.sum(wolpRewards, 0) / RUNS

    wolpRewards = np.transpose(wolpRewards)

    percentile5wolp = np.percentile(wolpRewards, 5, axis=1)
    percentile95wolp = np.percentile(wolpRewards, 95, axis=1)

    return wolpVal, percentile5wolp, percentile95wolp

def plotReinforce(ARMS, SCHEDULE, INTERVAL):

    reinforceRewards = []

    for i in range(RUNS):
        reinforceFileName = (f'../testResults/deadline_env/reinforceResults_arms_{ARMS}_batchSize_{BATCHSIZE}\
_lr_{REINFORCELR}_run_{i}_schedule_{SCHEDULE}.csv')
        df = pd.read_csv(reinforceFileName)
        runReward = list(df.iloc[:,1])
        reinforceRewards.append(runReward)

    reinforceVal = np.sum(reinforceRewards, 0) / RUNS
    reinforcePercentileRewards = np.transpose(reinforceRewards)

    percentile5Reinforce = np.percentile(reinforcePercentileRewards, 5, axis=1)
    percentile95Reinforce = np.percentile(reinforcePercentileRewards, 95, axis=1)

    return reinforceVal, percentile5Reinforce, percentile95Reinforce

def plotAQL(ARMS, SCHEDULE, INTERVAL):

    aqlRewards = []

    for i in range(RUNS):
        aqlFileName = (f'../testResults/deadline_env/aqlResults_arms_{ARMS}_run_{i}_schedule_{SCHEDULE}.csv')
        df = pd.read_csv(aqlFileName)
        runReward = df.iloc[:, 1]
        aqlRewards.append(runReward)

    aqlVal = np.sum(aqlRewards, 0) / RUNS

    aqlRewards = np.transpose(aqlRewards)

    percentile5aql = np.percentile(aqlRewards, 5, axis=1)
    percentile95aql = np.percentile(aqlRewards, 95, axis=1)

    return aqlVal, percentile5aql, percentile95aql 


def plotQWIC(ARMS, SCHEDULE, INTERVAL):


    qLearningRewards = []

    for i in range(RUNS):
        qLearningFileName = (f'../testResults/deadline_env/qLearningResults_arms_{ARMS}_run_{i}_schedule_{SCHEDULE}.csv')
        df = pd.read_csv(qLearningFileName)
        runReward = df.iloc[:, 1]
        qLearningRewards.append(runReward)

    qLearningVal = np.sum(qLearningRewards, 0) / RUNS
    qLearningRewards = np.transpose(qLearningRewards)

    qLearningpercentile5 = np.percentile(qLearningRewards, 5, axis=1)
    qLearningpercentile95 = np.percentile(qLearningRewards, 95, axis=1)

    return qLearningVal, qLearningpercentile5, qLearningpercentile95 

def plotWIBQL(ARMS, SCHEDULE, INTERVAL):

    wibqlRewards = []

    for i in range(RUNS):
        wibqlFileName = (f'../testResults/deadline_env/wibqlResults_arms_{ARMS}_run_{i}_schedule_{SCHEDULE}.csv')
        df = pd.read_csv(wibqlFileName)
        runReward = df.iloc[:, 1]
        wibqlRewards.append(runReward)

    wibqlVal = np.sum(wibqlRewards, 0) / RUNS
    wibqlRewards = np.transpose(wibqlRewards)

    wibqlpercentile5 = np.percentile(wibqlRewards, 5, axis=1)
    wibqlpercentile95 = np.percentile(wibqlRewards, 95, axis=1)

    return wibqlVal, wibqlpercentile5, wibqlpercentile95 


######################################################################################################################

fig, axes = plt.subplots(nrows=1,ncols=3, figsize=(WIDTH, HEIGHT), gridspec_kw={'wspace':0.13, 'hspace':0.0}, frameon=False)


for i in range(len(ARMSLIST)):
    ARMS = ARMSLIST[i]
    SCHEDULE = SCHEDULELIST[i]
    INTERVAL = INTERVALLIST[i]
    numEpisode = np.arange(0, EPISODEEND + INTERVAL, INTERVAL) 


    deadlineRewards, deadlinePercentile5, deadlinePercentile95 = plotDeadlineIndex(ARMS, SCHEDULE)
    NeurWINRewards, NeurWINPercentile5, NeurWINPercentile95 = plotNeurWIN(ARMS, SCHEDULE, INTERVAL)
    wolpRewards, wolpPercentile5, wolpPercentile95 = plotWolp(ARMS, SCHEDULE, INTERVAL)
    aqlRewards, aqlPercentile5, aqlPercentile95 = plotAQL(ARMS, SCHEDULE, INTERVAL)
    qwicRewards, qwicPercentile5, qwicPercentile95 = plotQWIC(ARMS, SCHEDULE, INTERVAL)
    wibqlRewards, wibqlPercentile5, wibqlPercentile95 = plotWIBQL(ARMS, SCHEDULE, INTERVAL)


    axes[i].hlines(xmin=0, xmax=EPISODEEND, y=deadlineRewards, label='Deadline Index', color='r', linewidth=LINEWIDTH, linestyle='dashdot', zorder=4)
    axes[i].fill_between(x=numEpisode, y1=deadlinePercentile5, y2=deadlinePercentile95, alpha=0.2, color='orange')
    axes[i].plot(numEpisode, NeurWINRewards, label='NeurWIN', linewidth=LINEWIDTH, linestyle='solid', zorder=5)
    axes[i].fill_between(x=numEpisode, y1=NeurWINPercentile5, y2=NeurWINPercentile95, alpha=0.2, color='green')
    axes[i].plot(numEpisode, aqlRewards, label=f'AQL', color='saddlebrown', linewidth=LINEWIDTH, linestyle='dotted')

    axes[i].fill_between(x=numEpisode, y1=aqlPercentile5, y2=aqlPercentile95, alpha=0.2, color='saddlebrown')
    axes[i].plot(numEpisode, qwicRewards, label=f'QWIC', color='g', linewidth=LINEWIDTH, linestyle=(0, (3,1,1,1,1,1)))
    axes[i].fill_between(x=numEpisode,y1=qwicPercentile5, y2=qwicPercentile95, alpha=0.2, color='teal')
    axes[i].plot(numEpisode, wibqlRewards, label=f'WIBQL', color='lightseagreen', linewidth=LINEWIDTH, linestyle=(0, (3,1,3,3,1,3)))
    axes[i].fill_between(x=numEpisode,y1=wibqlPercentile5, y2=wibqlPercentile95, alpha=0.2, color='lightseagreen')

    if ARMS == 100:
        pass
    else:
        reinforceRewards, reinforcePercentile5, reinforcePercentile95 = plotReinforce(ARMS, SCHEDULE, INTERVAL)
        axes[i].plot(numEpisode, reinforceRewards, label=f'REINFORCE', color='k', linewidth=LINEWIDTH, linestyle='dotted')
        axes[i].fill_between(x=numEpisode, y1=reinforcePercentile5, y2=reinforcePercentile95, alpha=0.2, color='k')
        axes[i].plot(numEpisode, wolpRewards, label='WOLP-DDPG', color='darkorchid', linewidth=LINEWIDTH, linestyle='dashed')
        axes[i].fill_between(x=numEpisode, y1=wolpPercentile5, y2=wolpPercentile95, alpha=0.2, color='darkorchid')
    axes[i].tick_params(axis='y', rotation=90)
    axes[i].set_xticks(np.arange(0,EPISODEEND+1,500))
    if ARMS == 100:
        pass 
    else:
        handles, labels = axes[i].get_legend_handles_labels()
    yStart, yEnd = axes[i].get_ylim()
    
    yLimits = np.linspace(yStart, yEnd, 4)
    yTicks = [50*round(num/50) for num in yLimits]
    axes[i].set_yticks(yTicks)


plt.legend(handles, labels, frameon=False, bbox_to_anchor=(1,0.95))


axes[0].set_title(f'N = 4     M = 1', weight='bold', fontsize=14)
axes[1].set_title(f'N = 10     M = 1', weight='bold', fontsize=14)
axes[2].set_title(f'N = 100     M = 25', weight='bold', fontsize=14)

axes[0].set_ylabel('Total Discounted Rewards', weight='bold')#, fontsize=15)
axes[1].set_xlabel('Training Episodes', weight='bold')#, fontsize=15)
plt.savefig('../plotResults/deadline_results/deadline_rewards.pdf', bbox_inches='tight')
plt.show()











