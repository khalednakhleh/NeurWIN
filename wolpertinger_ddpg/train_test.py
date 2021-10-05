#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import random 
import torch 
import sys 
import scipy.special
import itertools
sys.path.insert(0,'../')
from envs.recoveringBanditsEnv import recoveringBanditsEnv
from envs.deadlineSchedulingEnv import deadlineSchedulingEnv
from envs.sizeAwareIndexEnv import sizeAwareIndexEnv
import pandas as pd 

def train(continuous, env, agent, max_episode, warmup, save_model_dir, max_episode_length, logger, saveInterval):
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    s_t = None
    episodeRanges = np.arange(0, max_episode+saveInterval, saveInterval)

    agent.save_model(save_model_dir, episode=0)
    while episode < max_episode:
        print(f'current episode: {episode+1}')
        while True:
            if s_t is None:
                s_t = env.reset()
                agent.reset(s_t)
                print(s_t)

            if step <= warmup:
                action = agent.random_action()
            else:
                action = agent.select_action(s_t)
           
            if not continuous:
                action = action.reshape(1,).astype(int)[0]
            
            s_t1, r_t, done, _ = env.step(action)

            agent.observe(r_t, s_t1, done)
            if step > warmup:
                agent.update_policy()
            
            step += 1
            episode_steps += 1
            episode_reward += r_t
            s_t = s_t1

            if done:  
                
                logger.info(
                    "Ep:{0} | R:{1:.4f}".format(episode+1, episode_reward)
                )

                agent.memory.append(
                    s_t,
                    agent.select_action(s_t),
                    0., True
                )
                
                s_t = None
                episode_steps =  0
                episode_reward = 0.
                episode += 1
                
                break
        
        if episode in episodeRanges:
            agent.save_model(save_model_dir, episode)
            logger.info("### Model Saved before Ep:{0} ###".format(episode+1))

def initializeRecovering(envSeeds, TIMELIMIT, episodeEnd, theta, arms): 
    envs = {}

    for i in range(arms):
        env = recoveringBanditsEnv(seed=envSeeds[i], numEpisodes=1, episodeLimit=TIMELIMIT, train=False, 
batchSize=episodeEnd, thetaVals=theta[i], noiseVar=0.0, maxWait = 20)        
        envs[i] = env

    return envs 

def initializeSizeAware(envSeeds, TIMELIMIT,episodeEnd,arms, TESTLOAD1, TESTLOAD2, args):
    numClass1 = numClass2 = int(arms/2)
    num1 = numClass1
    num2 = numClass2
    TESTLOAD1 = TESTLOAD1
    TESTLOAD2 = TESTLOAD2
    GOODPROB1 = 0.75
    GOODPROB2 = 0.1
    load1Index = 0
    load2Index = 0
    envs = {}
    for i in range(arms):
        if num1 != 0:
            env = sizeAwareIndexEnv(numEpisodes=1, HOLDINGCOST=1, seed=envSeeds[i], Training=False,
            r1=8400, r2=33600, q=GOODPROB1, case=1, classVal=1, load=TESTLOAD1[load1Index], noiseVar = 0.0,
            maxLoad = 1000000, batchSize=args.test_episode, episodeLimit=1000000, fixedSizeMDP=False)
            load1Index += 1
            num1 -= 1
        elif num2 != 0:
            env = sizeAwareIndexEnv(numEpisodes=1, HOLDINGCOST=1, seed=envSeeds[i], Training=False,
            r1=8400, r2=33600, q=GOODPROB2, case=1, classVal=2, load=TESTLOAD2[load2Index], noiseVar = 0.0,
            maxLoad = 1000000, batchSize=args.test_episode, episodeLimit=1000000, fixedSizeMDP=False)
            load2Index += 1
            num2 -= 1

        envs[i] = env
    return envs 

def initializeDeadline(envSeeds,TIMELIMIT,episodeEnd, arms): 
    envs = {}
    jobProbs = 0.7

    for i in range(arms):
        env = deadlineSchedulingEnv(seed=envSeeds[i], numEpisodes=1, episodeLimit=TIMELIMIT, maxDeadline=12,
maxLoad=9, newJobProb=jobProbs, processingCost=0.5, train=False, batchSize=episodeEnd, noiseVar=0)
        envs[i] = env

    return envs


def resetMultiDimEnv(envs):
    state = []
    for key in envs:
        vals = envs[key].reset()
        val1 = vals[0]
        val2 = vals[1]
        state.append(val1)
        state.append(val2)
    state = np.array(state, dtype=np.float32)
    return state

def resetRecovering(envs):
    state = []
    for key in envs:
        val = envs[key].reset()[0]
        state.append(val)
    state = np.array(state, dtype=np.float32)
    return state

def getActionTableLength(arms,schedule):

    scheduleArms = schedule
    ARMS = arms 
    actionTable = np.zeros(int(scipy.special.binom(ARMS, scheduleArms)))
    n = int(ARMS)
    actionTable  = list(itertools.product([0, 1], repeat=n))
    actionTable = [x for x in actionTable if not sum(x) != scheduleArms]
    
    return actionTable


def test(agent, SEED, RUNS, test_episode, testing_interval, max_episode_length, logger, args):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    TESTLOAD1 = np.random.randint(1, 1000000, size=args.arms)
    TESTLOAD2 = np.random.randint(1, 1000000, size=args.arms)
    G = np.random.RandomState(args.seed)

    BETA = args.gamma 
    ARMS = args.arms 
    scheduleArms = args.scheduleArms
    actionTable = getActionTableLength(arms=ARMS, schedule=scheduleArms)
    

    for r in range(RUNS):
        total_reward = []
        envSeeds = G.randint(0, 10000, size=args.arms)
        
        testing_episodes = np.arange(0, test_episode+testing_interval, testing_interval)

        agent.is_training = False
        agent.eval()
        
        for episode in testing_episodes:
            state = []
            envs = {}
            time = 0
            rewards = []
            
            if args.env == 'recovering':
                classATheta = [10., 0.2, 0.0]
                classBTheta = [8.5, 0.4, 0.0]
                classCTheta = [7., 0.6, 0.0]
                classDTheta = [5.5, 0.8, 0.0]
                
                THETA = [classATheta, classBTheta, classCTheta, classDTheta, classATheta, classBTheta, classCTheta, classDTheta, classATheta, classBTheta]
                envs = initializeRecovering(envSeeds, TIMELIMIT=args.max_episode_length, episodeEnd = args.test_episode, theta=THETA,arms=args.arms)
                state = resetRecovering(envs)
                agent.load_weights(f'recovering_arms_{args.arms}_schedule_{args.scheduleArms}', episode)
                envId = 1

                print(f'testing recovering')
            elif args.env == 'deadline':
                envId = 2
                envs = initializeDeadline(envSeeds, TIMELIMIT=args.max_episode_length, episodeEnd=args.test_episode,arms=args.arms)
                state = resetMultiDimEnv(envs)
                
               
                agent.load_weights(f'deadline_arms_{args.arms}_schedule_{args.scheduleArms}', episode)
                print(f'testing deadline')
            elif args.env == 'size_aware':
                envId = 3
                envs = initializeSizeAware(envSeeds, TIMELIMIT=args.max_episode_length, episodeEnd=args.test_episode, 
                    arms=args.arms, TESTLOAD1=TESTLOAD1, TESTLOAD2=TESTLOAD2, args=args)
                state = resetMultiDimEnv(envs)
                agent.load_weights(f'size_aware_arms_{args.arms}_schedule_{args.scheduleArms}', episode)
                print(f'testing wireless scheduling')

            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            
            while True:
                action = policy(state)[0]
                actionVector = actionTable[action]
                
                stateVals = []

                cumReward = 0
                for i in range(len(actionVector)):
                    if actionVector[i] == 1:
                        nextState, reward, done, info = envs[i].step(1)
                        if len(nextState) > 1: 
                            stateVals.append(nextState[0])
                            stateVals.append(nextState[1])
                            if envId == 2:
                                cumReward += reward
                            elif envId == 3 and nextState[0] != 0.:
                                cumReward += reward
                        else: 
                            stateVals.append(nextState[0])
                            cumReward += reward
                    else:
                        nextState, reward, done, info = envs[i].step(0)
                        if len(nextState) > 1: 

                            stateVals.append(nextState[0])
                            stateVals.append(nextState[1])
                            if envId == 2:
                                cumReward += reward
                            elif envId == 3 and nextState[0] != 0.:
                                cumReward += reward
                        else: 
                            stateVals.append(nextState[0])
                            cumReward += reward 

                state = stateVals
                state = np.array(state, dtype=np.float32)

                rewards.append((BETA**time)*cumReward)
                time += 1
                
                if time == args.max_episode_length:
                    break

            total_reward.append((np.cumsum(rewards))[-1])
            print(f'finished for trained episode: {episode}. total rewards: {total_reward[-1]}')

        data = {'episode': np.arange(0, test_episode+testing_interval, testing_interval), 'cumulative_reward':total_reward}
        df = pd.DataFrame(data=data)
        ddpgFileName = (f'testResults/{args.env}_ddpg_results_{args.arms}_arms_choose_{args.scheduleArms}_run_{r}.csv')
        df.to_csv(ddpgFileName, index=False)
        print(f'finished DDPG scheduling for run {r+1}')