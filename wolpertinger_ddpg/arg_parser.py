#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

def init_parser(alg):

    if alg == 'WOLP_DDPG':

        parser = argparse.ArgumentParser(description='WOLP_DDPG')

        parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for rewards (default: 0.99)')
        parser.add_argument('--load', default=False, metavar='L', help='load a trained model')
        parser.add_argument('--load-model-dir', default='', type=str, help='folder to load trained models from')
        parser.add_argument('--gpu-ids', type=int, default=[1], nargs='+', help='GPUs to use [-1 CPU only]')
        parser.add_argument('--gpu-nums', type=int, default=1, help='#GPUs to use (default: 1)')
        parser.add_argument('--max-episode', type=int, default=50000, help='maximum #episode.')
        parser.add_argument('--test-episode', type=int, default=1000, help='maximum testing #episode.')
        parser.add_argument('--max-actions', default=200000, type=int, help='# max actions')
        parser.add_argument('--id', default='0', type=str, help='experiment id')
        parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
        parser.add_argument('--env', default='Pendulum-v0', type=str, help='Ride sharing')
        parser.add_argument('--hidden1', default=256, type=int, help='hidden num of first fully connect layer')
        parser.add_argument('--hidden2', default=128, type=int, help='hidden num of second fully connect layer')
        parser.add_argument('--c-lr', default=0.001, type=float, help='critic net learning rate')
        parser.add_argument('--p-lr', default=0.001, type=float, help='policy net learning rate (only for DDPG)')
        parser.add_argument('--warmup', default=128, type=int, help='time without training but only filling the replay memory')
        parser.add_argument('--bsize', default=64, type=int, help='batch size of transitions')
        parser.add_argument('--rmsize', default=200000, type=int, help='memory size')
        parser.add_argument('--window_length', default=1, type=int, help='')
        parser.add_argument('--tau-update', default=0.001, type=float, help='moving average for target network')
        parser.add_argument('--ou_theta', default=0.0, type=float, help='noise theta')
        parser.add_argument('--ou_sigma', default=0.0, type=float, help='noise sigma')
        parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
        parser.add_argument('--max_episode_length', default=500, type=int, help='maximum episode length')
        parser.add_argument('--init_w', default=0.003, type=float, help='')
        parser.add_argument('--epsilon', default=80000, type=int, help='Linear decay of exploration policy')
        parser.add_argument('--seed', default=-1, type=int, help='')
        parser.add_argument('--weight-decay', default=0.0001, type=float, help='weight decay for L2 Regularization loss')
        parser.add_argument('--save_episode_interval', default=5, type=int, help='when to save trained model')
        parser.add_argument('--env_seed', default=30, type=int, help='')
        parser.add_argument('--k_ratio', default=1, type=float,help='')
        parser.add_argument('--testing_episode_interval', default='100', type=int, help='interval between trained episodes')
        parser.add_argument('--test_runs', default='50', type=int, help='number of independent runs to test on')
        parser.add_argument('--scheduleArms', default ='1', type=int, help='how many arms to schedule in one timestep')
        parser.add_argument('--arms', type=int, help='number of arms that form the fixed MDP')
        #parser.add_argument('--update_every_episode', type=int, help='perform gradient step every n episodes')

        return parser

    else:

        raise RuntimeError('undefined algorithm {}'.format(alg))