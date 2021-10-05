
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
INTERVALLIST = [100, 100, 100]
CASES = [1,2,5]



########################## CONSTANTS ######################

LINEWIDTH = 1.75
REINFORCELR = 0.001
TIMELIMIT = 300
BATCHSIZE = 5
EPISODEEND = 30000
BASES = [20, 50, 100]


######################################################################################################################

fig, axes = plt.subplots(nrows=1,ncols=3, figsize=(WIDTH, HEIGHT), gridspec_kw={'wspace':0.14, 'hspace':0.0}, frameon=False)



for i in range(len(ARMSLIST)):

    ARMS = ARMSLIST[i]
    SCHEDULE = SCHEDULELIST[i]
    INTERVAL = INTERVALLIST[i]
    CASE = CASES[i]

    numEpisode = np.arange(0, EPISODEEND + INTERVAL, INTERVAL)

    neurwinFileName = (f'../testResults/recovering_env/nnIndexResults_arms_{ARMS}_batchSize_{BATCHSIZE}_case_{CASE}_timeLimit_{TIMELIMIT}_schedule_{SCHEDULE}.csv')
    aqlFileName = (f'../testResults/recovering_env/aqlResults_arms_{ARMS}_run_{0}_schedule_{SCHEDULE}.csv')
    qwicFileName = (f'../testResults/recovering_env/qLearningResults_arms_{ARMS}_run_{0}_schedule_{SCHEDULE}.csv')
    wibqlFileName = (f'../testResults/recovering_env/wibqlResults_arms_{ARMS}_run_{0}_schedule_{SCHEDULE}.csv')
    
    tsd1FileName = (f'../testResults/recovering_env/tsResults_arms_{ARMS}_timeLimit_{TIMELIMIT}_d_1_case_{CASE}_schedule_{SCHEDULE}.csv')
    ucbd1FileName = (f'../testResults/recovering_env/ucbResults_arms_{ARMS}_timeLimit_{TIMELIMIT}_d_1_case_{CASE}_schedule_{SCHEDULE}.csv')


    df = pd.read_csv(tsd1FileName)
    tsd1Rewards = df.iloc[:,1]

    df = pd.read_csv(ucbd1FileName)
    ucbd1Rewards = df.iloc[:,1]

    df = pd.read_csv(neurwinFileName)
    NeurWINRewards = df.iloc[:, 1]

    df = pd.read_csv(aqlFileName)
    aqlRewards = df.iloc[:, 1]

    df = pd.read_csv(qwicFileName)
    qwicRewards = df.iloc[:,1]

    df = pd.read_csv(wibqlFileName)
    wibqlRewards = df.iloc[:,1]


    d20LookAheadFileName = (f'../testResults/recovering_env/dLookAheadResults_arms_{ARMS}_timeLimit_{TIMELIMIT}_d_{20}_case_{CASE}_schedule_{SCHEDULE}.csv')
    df = pd.read_csv(d20LookAheadFileName)
    d20rewards = list(df.iloc[:,1])


    axes[i].plot(numEpisode, NeurWINRewards[:301], label='NeurWIN', color = 'b', linewidth=LINEWIDTH, linestyle='solid', zorder=5)
    axes[i].plot(numEpisode, aqlRewards, label='AQL', color='saddlebrown',linewidth=LINEWIDTH, linestyle='dotted')
    axes[i].plot(numEpisode, qwicRewards[:301], label='QWIC', color='g', linewidth=LINEWIDTH, linestyle=(0, (3,1,1,1,1,1)))
    axes[i].plot(numEpisode, wibqlRewards, label='WIBQL', color='lightseagreen', linewidth=LINEWIDTH, linestyle=(0, (3,1,3,3,1,3)))
    axes[i].plot(numEpisode, ucbd1Rewards, label='RGP-UCB. d=1', linewidth=LINEWIDTH, color='seagreen')
    axes[i].plot(numEpisode, tsd1Rewards, label='RGP-TS. d=1', linewidth=LINEWIDTH, color='chocolate')
    
    axes[i].hlines(xmin=0, xmax=EPISODEEND, y=d20rewards, label='Oracle, d = 20', color='maroon', linewidth=LINEWIDTH, linestyle='dashdot')

    if ARMS != 100:

        wolpFileName = (f'../wolpertinger_ddpg/testResults/recovering_ddpg_results_{ARMS}_arms_choose_{SCHEDULE}_run_{0}.csv')
        reinforceFileName = (f'../testResults/recovering_env/reinforceResults_arms_{ARMS}_batchSize_{BATCHSIZE}_lr_{REINFORCELR}_run_{0}_schedule_{SCHEDULE}.csv')

        df = pd.read_csv(wolpFileName)
        wolpRewards = df.iloc[:, 1]
        df = pd.read_csv(reinforceFileName)
        reinforceRewards = df.iloc[:,1]

        axes[i].plot(numEpisode, reinforceRewards, label='REINFORCE', color='k', linewidth=LINEWIDTH, linestyle='dotted')
        axes[i].plot(numEpisode, wolpRewards, label='WOLP-DDPG', color='darkorchid',linewidth=LINEWIDTH, linestyle='dashed')
        handles, labels = axes[i].get_legend_handles_labels()

    axes[i].tick_params(axis='y', rotation=90)
    axes[i].set_xticks(np.arange(0,EPISODEEND+1, 10000))

    yStart, yEnd = axes[i].get_ylim()
    yLimits = np.linspace(yStart, yEnd, 4)
    yTicks = [BASES[i]*round(num/BASES[i]) for num in yLimits]
    axes[i].set_yticks(yTicks)


plt.legend(handles, labels, frameon=False, bbox_to_anchor=(1.75,1.1))


axes[0].set_title(f'N = 4     M = 1', weight='bold', fontsize=14)
axes[1].set_title(f'N = 10     M = 1', weight='bold', fontsize=14)
axes[2].set_title(f'N = 100     M = 25', weight='bold', fontsize=14)

axes[0].set_ylabel('Total Discounted Rewards', weight='bold')
axes[1].set_xlabel('Training Episodes', weight='bold')


plt.savefig(f'../plotResults/recovering_results/recovering_results.pdf', bbox_inches='tight')
plt.show()


######################### PLOTTING THE RECOVERING FUNCTIONS ##################################################

'''
maxWait = 20
STYLES = ['solid','dashed','dotted','dashdot']
LABELS = ['A','B','C','D']
THETAS = [[10., 0.2, 0.0],[8.5, 0.4, 0.0],[7., 0.6, 0.0],[5.5, 0.8, 0.0]]

for x in range(len(THETAS)):
    rewards = []
    THETA = THETAS[x]

    for i in range(1,maxWait+1):

        reward = THETA[0] * (1 - np.exp(-1*THETA[1] * i + THETA[2]))
        rewards.append(reward)
    plt.plot(range(1,maxWait+1), rewards, linewidth=LINEWIDTH, label=f'Recovering function {LABELS[x]}',linestyle=STYLES[x])

plt.ylabel('$f(z)$', fontsize=14)
plt.xlabel('$z \in \{1, z_{max}\}$', fontsize=14)
plt.legend(frameon=False)
plt.xticks(np.arange(1,maxWait+1))
plt.savefig(f'../plotResults/recovering_results/recovering_functions.pdf')
plt.show()
'''

