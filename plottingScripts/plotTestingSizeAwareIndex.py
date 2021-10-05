
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

SCHEDULELIST = [1, 1, 25]     
ARMSLIST = [4, 10, 100]       
INTERVALLIST = [1000, 1000, 1000]

############################# CONSTANTS ##########################
LINEWIDTH = 1.75 
BATCHSIZE = 5 
BETA = 0.99
EPISODEEND = 30000
RUNS = 50
TIMELIMIT = 300 
REINFORCELR = 0.001
CASE = 1 

bases = [5, 20, 60]
numberYLimits = [5, 4, 4]


def plotSizeAware(ARMS, SCHEDULE, INTERVAL):

    sizeAwareFileName = (f'../testResults/size_aware_env/case_{CASE}/sizeAwareIndexResults_arms_{ARMS}_schedule_{SCHEDULE}_arms.csv')
    df = pd.read_csv(sizeAwareFileName)
    sizeAwareRewards = df.iloc[:,1]

    sizeAware5Percentile, sizeAware95Percentile = np.percentile(sizeAwareRewards[0:RUNS], [5, 95])
    sizeAwareRewards = sum(sizeAwareRewards[0:RUNS]) / RUNS

    return sizeAwareRewards, sizeAware5Percentile, sizeAware95Percentile


def plotNeurWIN(ARMS, SCHEDULE, INTERVAL):


    nnRewards = []

    for i in range(RUNS):
        NeurWINFileName = (f'../testResults/size_aware_env/case_{CASE}/nnIndexResults_arms_{ARMS}_batchSize_{BATCHSIZE}_run_{i}_schedule_{SCHEDULE}_arms.csv')
        df = pd.read_csv(NeurWINFileName)
        runReward = df.iloc[:, 1]
        nnRewards.append(runReward)

    nnRewards = np.array(nnRewards)
    nnVal = np.sum(nnRewards, 0) / RUNS


    nnRewards = np.transpose(nnRewards)


    percentile5 = np.percentile(nnRewards, 5, axis=1)
    percentile95 = np.percentile(nnRewards, 95, axis=1)
    
    return nnVal, percentile5, percentile95


def plotWolp(ARMS, SCHEDULE, INTERVAL):

    wolpRewards = []

    for i in range(RUNS):
        if ARMS == 100:
            wolpFileName = (f'../testResults/size_aware_env/case_{CASE}/wolpResults_arms_{ARMS}_batchSize_{BATCHSIZE}_run_{i}_timeLimit_{TIMELIMIT}_schedule_{SCHEDULE}.csv')
        else: 
            wolpFileName = (f'../wolpertinger_ddpg/testResults/size_aware_ddpg_results_{ARMS}_arms_choose_{SCHEDULE}_run_{i}.csv')
        df = pd.read_csv(wolpFileName)
        runReward = df.iloc[:, 1]
        wolpRewards.append(runReward)

    wolpVal = np.sum(wolpRewards, 0) / RUNS

    wolpRewards = np.transpose(wolpRewards)

    percentile5wolp = np.percentile(wolpRewards, 5, axis=1)
    percentile95wolp = np.percentile(wolpRewards, 95, axis=1)

    return wolpVal, percentile5wolp, percentile95wolp 

def plotAQL(ARMS, SCHEDULE, INTERVAL):


    aqlRewards = []

    for i in range(RUNS):
        aqlFileName = (f'../testResults/size_aware_env/case_{CASE}/aqlResults_arms_{ARMS}_run_{i}_schedule_{SCHEDULE}.csv')
        df = pd.read_csv(aqlFileName)
        runReward = df.iloc[:, 1]
        aqlRewards.append(runReward)

    aqlVal = np.sum(aqlRewards, 0) / RUNS

    aqlRewards = np.transpose(aqlRewards)

    percentile5aql = np.percentile(aqlRewards, 5, axis=1)
    percentile95aql = np.percentile(aqlRewards, 95, axis=1)

    return aqlVal, percentile5aql, percentile95aql


def plotReinforce(ARMS, SCHEDULE, INTERVAL):

    reinforceRewards = []


    for i in range(RUNS):
        reinforceFileName = (f'../testResults/size_aware_env/case_{CASE}/reinforceResults_arms_{ARMS}\
_batchSize_{BATCHSIZE}_lr_{REINFORCELR}_run_{i}_schedule_{SCHEDULE}.csv')
        df = pd.read_csv(reinforceFileName)
        runCost = df.iloc[:, 1] 
        reinforceRewards.append(runCost)


    reinforcePercentileRewards = np.transpose(reinforceRewards)
    reinforceVal = sum(reinforceRewards)  / RUNS

    percentile5Reinforce = np.percentile(reinforcePercentileRewards, 5, axis=1)
    percentile95Reinforce = np.percentile(reinforcePercentileRewards, 95, axis=1)

    return reinforceVal, percentile5Reinforce, percentile95Reinforce
##################################################################


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(WIDTH, HEIGHT), gridspec_kw={'wspace':0.14, 'hspace':0.0}, frameon=False)



for i in range(len(ARMSLIST)):
    ARMS = ARMSLIST[i]
    SCHEDULE = SCHEDULELIST[i]
    INTERVAL = INTERVALLIST[i]
    numEpisode = np.arange(0, EPISODEEND + INTERVAL, INTERVAL) 


    sizeAwareRewards, sizeAwarePercentile5, sizeAwarePercentile95 = plotSizeAware(ARMS, SCHEDULE, INTERVAL)
    axes[i].hlines(xmin=0, xmax=EPISODEEND, y=sizeAwareRewards, label='Size Aware Index', color='r',linewidth=LINEWIDTH, linestyle='dashdot', zorder=4)
    NeurWINRewards, NeurWINPercentile5, NeurWINPercentile95 = plotNeurWIN(ARMS, SCHEDULE, INTERVAL)
    axes[i].plot(numEpisode, NeurWINRewards[:31], label='NeurWIN', color='b', linewidth=LINEWIDTH, linestyle='solid',zorder=5)

    if ARMS == 100:
        pass
    else:
        reinforceRewards, reinforcePercentile5, reinforcePercentile95 = plotReinforce(ARMS, SCHEDULE, INTERVAL)
        wolpRewards, wolpPercentile5, wolpPercentile95 = plotWolp(ARMS, SCHEDULE, INTERVAL)


    aqlRewards, aqlPercentile5, aqlPercentile95 = plotAQL(ARMS, SCHEDULE, INTERVAL)

    axes[i].plot(numEpisode, aqlRewards, label='AQL', color='saddlebrown',linewidth=LINEWIDTH, linestyle='dotted')
    axes[i].fill_between(x=numEpisode, y1=aqlPercentile5, y2=aqlPercentile95, alpha=0.2, color='saddlebrown')

    axes[i].fill_between(x=numEpisode,y1=sizeAwarePercentile5, y2=sizeAwarePercentile95, alpha=0.2, color='orange')
    axes[i].fill_between(x=numEpisode,y1=NeurWINPercentile5[:31], y2=NeurWINPercentile95[:31], alpha=0.2, color='green')

    if ARMS == 100:
        pass 
    else:
        axes[i].plot(numEpisode, reinforceRewards, label='REINFORCE', color='k',linewidth=LINEWIDTH, linestyle='dotted')
        axes[i].fill_between(x=numEpisode, y1=reinforcePercentile5, y2=reinforcePercentile95, alpha=0.2, color='k')

        axes[i].plot(numEpisode, wolpRewards, label='WOLP-DDPG', color='darkorchid', linewidth=LINEWIDTH, linestyle='dashed')
        axes[i].fill_between(x=numEpisode, y1=wolpPercentile5, y2=wolpPercentile95, alpha=0.2, color='darkorchid')

    axes[i].tick_params(axis='y', rotation=90)
    axes[i].set_xticks(np.arange(0,EPISODEEND+1, 10000))


    if ARMS == 100:
        pass 
    else: 
        handles, labels = axes[i].get_legend_handles_labels()

    yStart, yEnd = axes[i].get_ylim()
    yLimits = np.linspace(yStart, yEnd, numberYLimits[i])
    yTicks = [bases[i]*round(num/bases[i]) for num in yLimits]
    axes[i].set_yticks(yTicks)



plt.legend(handles, labels, frameon=False, bbox_to_anchor=(1.8,0.9))

axes[0].set_title(f'N = 4    M = 1', weight='bold', fontsize=14)
axes[1].set_title(f'N = 10    M = 1', weight='bold', fontsize=14)
axes[2].set_title(f'N = 100    M = 25', weight='bold', fontsize=14)


axes[0].set_ylabel('Total Discounted Rewards', weight='bold')
axes[1].set_xlabel('Training Episodes', weight='bold')

plt.savefig('../plotResults/size_aware_results/size_aware_rewards.pdf', bbox_inches='tight')
plt.show()







