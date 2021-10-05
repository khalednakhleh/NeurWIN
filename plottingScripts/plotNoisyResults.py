

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


########################## CONSTANTS ######################

NOISEVARS = [0.0, 0.01, 0.04, 0.16]
ARMS = 100
SCHEDULE = 25
LINEWIDTH = 1.75
RUNS = 50 
BATCHSIZE = 5
TIMELIMIT = 300

DEADLINEEPISODEEND = 2000
RECOVERINGEPISODEEND = 30000
WIRELESSEPISODEEND = 30000 

fig, axes = plt.subplots(nrows=1,ncols=3, figsize=(WIDTH, HEIGHT), gridspec_kw={'wspace':0.15, 'hspace':0.0}, frameon=False)
numberYLimits = [4, 4, 4]
bases = [50, 100, 50]

colors = ['b','g','k','saddlebrown', 'slateblue']
linestyles = ['solid', 'dashed', 'dotted', 'dashdot', (0, (3,1,1,1,1,1))]


######################################## PLOTTING DEADLINE ##################################################################
INTERVAL = 20
numEpisode = np.arange(0, DEADLINEEPISODEEND+INTERVAL, INTERVAL)

def plotDeadlineIndex(ARMS, SCHEDULE):

    deadlineFileName = (f'../testResults/deadline_env/deadlineIndexResults_arms_{ARMS}_timeLimit_{TIMELIMIT}_schedule_{SCHEDULE}.csv')
    df = pd.read_csv(deadlineFileName)
    deadlineRewards = df.iloc[:,1]
    deadline5Percentile, deadline95Percentile = np.percentile(deadlineRewards[0:RUNS], [5, 95])

    deadlineRewards = sum(deadlineRewards[0:RUNS] / RUNS)
    
    return deadlineRewards, deadline5Percentile, deadline95Percentile


deadlineRewards, deadlinePercentile5, deadlinePercentile95 = plotDeadlineIndex(ARMS, SCHEDULE)

deadlineIndex = axes[0].hlines(xmin=0, xmax=DEADLINEEPISODEEND, y=deadlineRewards, label='Deadline Index', color='maroon', linewidth=LINEWIDTH, linestyle='dashdot')
axes[0].fill_between(x=numEpisode,y1=deadlinePercentile5, y2=deadlinePercentile95, alpha=0.2, color='orange')#, label='Deadline Whittle Index confidence bound')


x = 0
for var in NOISEVARS:
    nnRewards = []
    for i in range(RUNS):
        if var == 0.0:
            NeurWINFileName = (f'../testResults/deadline_env/nnIndexResults_arms_{ARMS}_batchSize_{BATCHSIZE}_run_{i}_timeLimit_{TIMELIMIT}_schedule_{SCHEDULE}.csv')
        else:
            NeurWINFileName = (f'../testResults/deadline_env/noisy_results/noise_val_{var}/nnIndexResults_arms_{ARMS}_batchSize_{BATCHSIZE}_run_{i}_timeLimit_{TIMELIMIT}_schedule_{SCHEDULE}.csv')

        df = pd.read_csv(NeurWINFileName)
        runReward = df.iloc[:, 1]
        nnRewards.append(runReward)

    nnVal = np.sum(nnRewards, 0) / RUNS

    nnRewards = np.transpose(nnRewards)

    percentile5 = np.percentile(nnRewards, 5, axis=1)
    percentile95 = np.percentile(nnRewards, 95, axis=1)
    #print(x)
    axes[0].plot(numEpisode, nnVal, label=f'NeurWIN {np.sqrt(var)*100:.0f}% error', color=colors[x], linewidth=LINEWIDTH, linestyle=linestyles[x])
    axes[0].fill_between(x=numEpisode,y1=percentile5, y2=percentile95, alpha=0.2, color=colors[x])#, label='NeurWIN confidence bound')

    x += 1

axes[0].set_ylim(-2800, -1900)

yStart, yEnd = axes[0].get_ylim()
yLimits = np.linspace(yStart, yEnd, numberYLimits[0])
yTicks = [bases[0]*round(num/bases[0]) for num in yLimits]
axes[0].set_yticks(yTicks)



#yStart, yEnd = axes[0].get_ylim()
axes[0].set_xticks(np.arange(0, DEADLINEEPISODEEND+1, 500))


######################################## PLOTTING RECOVERING ################################################################

INTERVAL = 100
CASE = 5
numEpisode = np.arange(0, RECOVERINGEPISODEEND+INTERVAL, INTERVAL)


d20LookAheadFileName = (f'../testResults/recovering_env/dLookAheadResults_arms_{ARMS}_timeLimit_{TIMELIMIT}_d_{20}_case_{CASE}_schedule_{SCHEDULE}.csv')
df = pd.read_csv(d20LookAheadFileName)
d20rewards = list(df.iloc[:,1])


d20Lookahead = axes[1].hlines(xmin=0, xmax=RECOVERINGEPISODEEND, y=d20rewards, label='Oracle, d = 20', color='orange', linewidth=LINEWIDTH, linestyle='dashdot')

x = 0
for var in NOISEVARS:
    if var == 0:
        neurwinFileName = (f'../testResults/recovering_env/nnIndexResults_arms_{ARMS}_batchSize_{BATCHSIZE}_case_{CASE}_timeLimit_{TIMELIMIT}_schedule_{SCHEDULE}.csv')
    else:
        neurwinFileName = (f'../testResults/recovering_env/noisy_results/noise_val_{var}/nnIndexResults_arms_{ARMS}_batchSize_{BATCHSIZE}_case_{CASE}_timeLimit_{TIMELIMIT}_schedule_{SCHEDULE}.csv')
    df = pd.read_csv(neurwinFileName)
    nnRewards = df.iloc[:, 1]
    axes[1].plot(numEpisode, nnRewards[:301], label=f'NeurWIN ${np.sqrt(var)*100:.0f}$% error', color = colors[x], linewidth=LINEWIDTH, linestyle=linestyles[x])

    x += 1

axes[1].set_ylim(13500,14500)

yStart, yEnd = axes[1].get_ylim()
yLimits = np.linspace(yStart, yEnd, numberYLimits[1])
yTicks = [bases[1]*round(num/bases[1]) for num in yLimits]
axes[1].set_yticks(yTicks)



######################################## PLOTTING SIZE AWARE ################################################################


INTERVAL = 1000
CASE = 1
numEpisode = np.arange(0, WIRELESSEPISODEEND+INTERVAL, INTERVAL)


sizeAwareFileName = (f'../testResults/size_aware_env/case_{CASE}/sizeAwareIndexResults_arms_{ARMS}_schedule_{SCHEDULE}_arms.csv')
df = pd.read_csv(sizeAwareFileName)
sizeAwareRewards = df.iloc[:,1]

sizeAware5Percentile, sizeAware95Percentile = np.percentile(sizeAwareRewards, [5, 95])
sizeAwareRewards = sum(sizeAwareRewards[0:RUNS]) / RUNS


sizeAwareIndex = axes[2].hlines(xmin=0, xmax=WIRELESSEPISODEEND, y=sizeAwareRewards, label='Size-aware Index', color='r', linewidth=LINEWIDTH, linestyle='dashdot')
axes[2].fill_between(x=numEpisode,y1=sizeAware5Percentile, y2=sizeAware95Percentile, alpha=0.2, color='orange')

x = 0
for var in NOISEVARS:
    nnRewards = []
    for i in range(RUNS):
        if var == 0.0:
            NeurWINFileName = (f'../testResults/size_aware_env/case_{CASE}/nnIndexResults_arms_{ARMS}_batchSize_{BATCHSIZE}_run_{i}_schedule_{SCHEDULE}_arms.csv')
        else:
            NeurWINFileName = (f'../testResults/size_aware_env/noisy_results/noise_val_{var}/case_{CASE}/nnIndexResults_arms_{ARMS}_batchSize_{BATCHSIZE}_run_{i}_schedule_{SCHEDULE}_arms.csv')

        df = pd.read_csv(NeurWINFileName)
        runReward = df.iloc[:, 1]
        nnRewards.append(runReward)

    nnRewards = np.array(nnRewards)
    nnVal = np.sum(nnRewards, 0) / RUNS

    nnRewards = np.transpose(nnRewards)

    percentile5 = np.percentile(nnRewards, 5, axis=1)
    percentile95 = np.percentile(nnRewards, 95, axis=1)


    axes[2].plot(numEpisode, nnVal[:31], label=f'NeurWIN {np.sqrt(var)*100:.0f}% error', color=colors[x], linewidth=LINEWIDTH, linestyle=linestyles[x])
    axes[2].fill_between(x=numEpisode,y1=percentile5[:31], y2=percentile95[:31], alpha=0.2, color=colors[x])#, label='NeurWIN Confidence Bound')

    x += 1
axes[2].set_ylim(-3800, -3300)

yStart, yEnd = axes[2].get_ylim()
yLimits = np.linspace(yStart, yEnd, numberYLimits[2])
yTicks = [bases[2]*round(num/bases[2]) for num in yLimits]
axes[2].set_yticks(yTicks)


#############################################################################################################################

handles, labels = axes[0].get_legend_handles_labels()

handles.append(d20Lookahead)
labels.append('Oracle, d = 20')
handles.append(sizeAwareIndex)
labels.append('Size-aware Index')


plt.legend(handles, labels, frameon=False, bbox_to_anchor=(1,1))


for i in range(len(axes)):
	axes[i].tick_params(axis='y', rotation=90)


axes[0].set_title(f'Deadline Scheduling', weight='bold', fontsize=14)
axes[1].set_title(f'Recovering Bandits', weight='bold', fontsize=14)
axes[2].set_title(f'Wireless Scheduling', weight='bold', fontsize=14)

axes[0].set_ylabel('Total Discounted Rewards', weight='bold')#, fontsize=15)
axes[1].set_xlabel('Training Episodes', weight='bold')#, fontsize=15)

plt.savefig(f'../plotResults/noisy_results.pdf', bbox_inches='tight')
plt.show()






