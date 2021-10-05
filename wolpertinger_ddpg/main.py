#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import logging
from train_test import train, test
import warnings
from arg_parser import init_parser
from setproctitle import setproctitle as ptitle
from normalized_env import NormalizedEnv
import gym
import sys 
sys.path.insert(0,'../envs/')
from recoveringBanditsEnv import recoveringBanditsEnv
from recoveringBanditsMultipleArmsEnv import recoveringBanditsMultipleArmsEnv
from deadlineSchedulingEnv import deadlineSchedulingEnv
from deadlineSchedulingMultipleArmsEnv import deadlineSchedulingMultipleArmsEnv
from sizeAwareIndexEnv import sizeAwareIndexEnv
from sizeAwareIndexMultipleArmsEnv import sizeAwareIndexMultipleArmsEnv

if __name__ == "__main__":
    ptitle('test_wolp')
    warnings.filterwarnings('ignore')
    parser = init_parser('WOLP_DDPG')
    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)[1:-1]

    from util import get_output_folder, setup_logger
    from wolp_agent import WolpertingerAgent

    args.save_model_dir = get_output_folder('output', args.env)
    
    if args.env == 'recovering':
        print('selected recovering')
        env = recoveringBanditsMultipleArmsEnv(seed=args.env_seed, numEpisodes=args.max_episode, batchSize=5,
        train = args.mode, numArms=args.arms, scheduleArms=args.scheduleArms, noiseVar=0.0, maxWait=20, episodeLimit=args.max_episode_length)
    elif args.env == 'deadline':
    	print('selected deadline')
    	env = deadlineSchedulingMultipleArmsEnv(seed=args.env_seed, numEpisodes=args.max_episode, batchSize=5, 
        train=args.mode, numArms=args.arms, processingCost=0.5, maxDeadline=12, maxLoad=9, newJobProb=0.7, 
        episodeLimit=args.max_episode_length, scheduleArms=args.scheduleArms, noiseVar=0.0)
    elif args.env == 'size_aware':
    	class1Arms = class2Arms = int(args.arms / 2)
    	env = sizeAwareIndexMultipleArmsEnv(seed=args.env_seed, numEpisodes=args.max_episode, train=args.mode, noiseVar=0,
        batchSize = 5, class1Arms=class1Arms, class2Arms=class2Arms, numArms=args.arms, scheduleArms=args.scheduleArms, 
        case=1, episodeLimit=args.max_episode_length)

    continuous = None
    try:
        # continuous action
        
        nb_states = env.state_space.shape[0]
        nb_actions = env.action_space.shape[0]
        action_high = env.action_space.high
        action_low = env.action_space.low
        continuous = True
        env = NormalizedEnv(env)
    except IndexError:
        # discrete action for 1 dimension
        
        nb_states = env.state_space.shape[0]
        nb_actions = 1 
        max_actions = env.action_space.n
        continuous = False

    if args.seed > 0:
        np.random.seed(args.seed)
        

    if continuous:
        agent_args = {
            'continuous':continuous,
            'max_actions':None,
            'action_low': action_low,
            'action_high': action_high,
            'nb_states': nb_states,
            'nb_actions': nb_actions,
            'k_ratio': args.k_ratio,
            'args': args,
        }
    else:
        agent_args = {
            'continuous':continuous,
            'max_actions':max_actions,
            'action_low': None,
            'action_high': None,
            'nb_states': nb_states,
            'nb_actions': nb_actions,
            'k_ratio': args.k_ratio,
            'args': args,
        }

    agent = WolpertingerAgent(**agent_args)


    if args.gpu_ids[0] >= 0 and args.gpu_nums > 0:
        agent.cuda_convert()

    # set logger, log args here
    log = {}
    if args.mode == 'train':
        setup_logger('RS_log', r'{}/RS_train_log'.format(args.save_model_dir))
    elif args.mode == 'test':
        setup_logger('RS_log', r'{}/RS_test_log'.format(args.save_model_dir))
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
    log['RS_log'] = logging.getLogger('RS_log')
    d_args = vars(args)
    for k in d_args.keys():
        log['RS_log'].info('{0}: {1}'.format(k, d_args[k]))

    if args.mode == 'train':

        train_args = {
            'continuous':continuous,
            'env': env,
            'agent': agent,
            'max_episode': args.max_episode,
            'warmup': args.warmup,
            'save_model_dir': args.save_model_dir,
            'max_episode_length': args.max_episode_length,
            'logger': log['RS_log'],
            'saveInterval' : args.save_episode_interval,
        }

        train(**train_args)

    elif args.mode == 'test':

        test_args = {
            'agent': agent,
            'RUNS' : args.test_runs,
            'test_episode':args.test_episode,
            'max_episode_length': args.max_episode_length,
            'testing_interval' : args.testing_episode_interval,
            'SEED': args.seed,
            'logger': log['RS_log'],
            'args' : args,
        }

        test(**test_args)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
