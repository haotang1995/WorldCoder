#!/usr/bin/env python
# coding=utf-8

import os.path as osp
import copy
import nltk
from gitinfo import get_git_info

from .utils import extract_code_blocks, remove_duplicate_code, abbr_repr, count_tokens_for_openai, remove_noncompilable_code_blocks, get_avoid_words

def guess_reward(
    reward_code, mission, llm,
    code_example_num=3,
    verbose=True,
):
    mission = str(mission)
    model_args = {'temperature': 0.0}
    if verbose:
        print(f'Guessing reward function for {mission}')

    mission_rewards = sorted(reward_code.items(), key=lambda x: mission_distance(x[0], mission))[:code_example_num]
    _mission_rewards = copy.deepcopy(mission_rewards)
    while len(_mission_rewards) > 0:
        sample_code_in_texts = '\n\n'.join([
            f'The reward function code for mission "{mission}" is:\n```\n{reward_code}\n```'
            for mission, reward_code in mission_rewards
        ])
        if count_tokens_for_openai(sample_code_in_texts) > 5120:
            _mission_rewards = _mission_rewards[:-1]
        else:
            break;

    prefix_code = '\n'.join([x[1] for x in mission_rewards])
    prefix_code = remove_duplicate_code(prefix_code)
    chat_history = [
        {'role': 'system', 'content': FIRST_SYSTEM_MESSAGE.format()},
        {'role': 'user', 'content': FIRST_MESSAGE.format(
            sample_code=sample_code_in_texts,
            mission=str(mission),
        )},
    ]
    if verbose:
        print('-'*20 + 'Guessing rewards: Prompts' + '-'*20)
        for chat in chat_history:
            print()
            print(chat['role'] + ':')
            print(chat['content'])
            print()

    model_args['logit_bias'] = get_avoid_words()
    with llm.track() as cb:
        with llm.track_new() as new_cb:
            gen = llm(chat_history, model_args=model_args)
            gen = gen.choices[0].message
    if verbose:
        print('*'*20 + 'Guessing code: Machine Reply' + '*'*20)
        print(gen.content)
        print(cb)

    code_blocks = extract_code_blocks(gen.content)
    code_blocks = remove_noncompilable_code_blocks(code_blocks, prefix=prefix_code)
    code = prefix_code + '\n' + '\n'.join(code_blocks)
    code = remove_duplicate_code(code)
    if verbose:
        print('\nGuessing code: Final code')
        print(code)

    success_flag = True
    chat_history.append(gen)
    final_outputs = {
        'chat_history': chat_history,
        'configurations': {
            'mission': mission,
            'mission_rewards': mission_rewards,
            'code_example_num': code_example_num,
            'gitinfo': get_git_info(),
            'filename': osp.abspath(__file__),
        },
        'costs': {k:v for k, v in cb.usage.items() if k != '_lock'},
        'new_costs': {k:v for k, v in new_cb.usage.items() if k != '_lock'},
        'code': code,
    }
    output = {
        'success_flag': success_flag,
        'final_outputs': final_outputs,
        'code': code,
    }
    return output

def mission_distance(mission1, mission2):
    mission1 = mission1.lower()
    mission2 = mission2.lower()
    mission1 = mission1.replace('?', '').replace('!', '').replace('.', '').replace(',', '').replace('\'', '')
    mission2 = mission2.replace('?', '').replace('!', '').replace('.', '').replace(',', '').replace('\'', '')
    mission1 = mission1.split(' ')
    mission2 = mission2.split(' ')
    mission1 = [w for w in mission1 if len(w) > 0]
    mission2 = [w for w in mission2 if len(w) > 0]
    return nltk.edit_distance(mission1, mission2) / max(len(mission1), len(mission2))

FIRST_SYSTEM_MESSAGE = '''
You are a robot exploring in an object-centric environment. Your goal is to model the logic of the world in python, specifically the reward function that maps (state, action, next_state) to (reward, done). You will to be given the mission for the environment you are going to act in, as well as a few sample code from the other environments. You need to implement the new reward function for the new environment you are going to act in. The new code needs to be directly runnable on (state, action, next_state) and return (reward, done) in python.
'''


FIRST_MESSAGE = '''
Here is a few sample code for the reward function in other environments. Please check them in detail and think about how to implement the reward function for mission "{mission}" in the new environment. The code needs to be directly runnable on (state, action, next_state) and return (reward, done) in python.

{sample_code}

Now, you have entered a new environment. It shows a mission "{mission}". Do you know what this mission means and how to implement it in a reward function? Analyze the behaviors of the reward function case by case. In what situations will it return a positive reward or not? In what situations will it return done=True or not? Why? Please implement the code following the template in the sample code. You must implement the ` reward_func` function as the main function to be called by the environment. The code needs to be directly runnable on (mission, state, action, next_state) and return (reward, done) in python.
'''
