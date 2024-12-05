#!/usr/bin/env python
# coding=utf-8

import argparse

from .minigrid_env import MiniGridEnv

ENV_CLASS = {
    'minigrid': MiniGridEnv,
}

def add_env_args(parser):
    parser.add_argument('--env', type=str, default='minigrid', help='the name of environment')
    parser.add_argument('--env_name_list', type=str, nargs='+', default=None)
    parser.add_argument('--no_curri', dest='curri', action='store_false', help='whether to use curriculum learning')
    parser.add_argument('--env_seed', type=int, default=None, help='the seed of environment')
def get_env_args(args):
    output_args = dict()
    output_args['env'] = args.env.lower()
    if args.env_name_list is None:
        if args.env == 'minigrid':
            output_args['env_name_list'] = ['MiniGrid-Empty-5x5-v0', 'MiniGrid-DoorKey-5x5-v0', 'MiniGrid-Unlock-v0', 'MiniGrid-Fetch-6x6-N2-v0', 'MiniGrid-UnlockPickup-v0', 'MiniGrid-BlockedUnlockPickup-v0']
        else:
            raise NotImplementedError
    else:
        output_args['env_name_list'] = args.env_name_list
    if not args.curri:
        output_args['env_name_list'] = output_args['env_name_list'][:1]
    output_args['env_seed'] = args.env_seed if args.env_seed is not None else args.seed
    return output_args
def get_env(args):
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    env = ENV_CLASS[args.env](
        env_name=args.env_name,
        seed=args.env_seed,
    )
    return env
