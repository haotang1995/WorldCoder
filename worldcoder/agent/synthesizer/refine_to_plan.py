#!/usr/bin/env python
# coding=utf-8

import os.path as osp
from gitinfo import get_git_info

from .utils import extract_code_blocks, remove_duplicate_code, abbr_repr, count_tokens_for_openai, get_avoid_words, remove_unused_code
from .reward_utils import experiences2text
from .evaluator import SepPlanEvaluator
from ..world_model.main import _extract_api

def refine_to_plan(transit_code, reward_code, failed_env_info, llm, verbose=True,):
    assert isinstance(transit_code, str), f'transit_code should be str, but got {transit_code}'
    assert isinstance(reward_code, str), f'reward_code should be str, but got {reward_code}'
    verbose_flag = verbose
    mission = failed_env_info['mission']
    state = failed_env_info['state']

    api, _transit_code, _reward_code = _extract_api(transit_code, reward_code,)
    code = api + '\n' + _transit_code + '\n' + _reward_code
    old_code = remove_duplicate_code(code)

    valid_actions = failed_env_info['state'].get_action_space()

    chat_history = [
        {'role': 'system', 'content': FIRST_SYSTEM_MESSAGE.format()},
        {'role': 'user', 'content': FIRST_MESSAGE.format(
            code = old_code,
            mission=str(mission),
            initial_state=str(state),
            criterion=CRITERION,
            valid_actions=valid_actions,
        )},
    ]
    if verbose_flag:
        print('-'*20 + 'Refining code to plan: Prompts' + '-'*20)
        for chat in chat_history:
            print()
            print(chat['role'] + ':')
            print(chat['content'])
            print()

    model_args = {'logit_bias': get_avoid_words(['class',])}
    with llm.track() as cb:
        with llm.track_new() as new_cb:
            gen = llm(chat_history, model_args=model_args,)
            gen = gen.choices[0].message
    if verbose_flag:
        print('*'*20 + 'Refining code to plan: Machine Reply' + '*'*20)
        print(gen.content)
        print(cb)

    evaluator = SepPlanEvaluator()
    code_blocks = extract_code_blocks(gen.content)
    while True:
        code = old_code + '\n' + '\n'.join(code_blocks)
        code = remove_duplicate_code(code)
        result = evaluator(code, code, failed_env_info,)
        if (result['compilation_error'] is not None) and len(code_blocks) > 1:
            if verbose_flag:
                print('Compilation Error:', result['compilation_error'],)
            code_blocks = code_blocks[:-1]
        else:
            break
    if verbose_flag:
        print('\nPlan Results:', abbr_repr(result))

    success_flag = result['success_flag']

    chat_history.append({'role': 'assistant', 'content': gen.content})
    final_outputs = {
        'chat_history': chat_history,
        'result': result,
        'configurations': {
            'transit_code': transit_code,
            'reward_code': reward_code,
            'failed_env_info': failed_env_info,
            'gitinfo': get_git_info(),
            'filename': osp.abspath(__file__),
        },
        'costs': {k:v for k,v in cb.usage.items() if k != '_lock'},
        'new_costs': {k:v for k,v in new_cb.usage.items() if k != '_lock'},
        'code': code,
    }
    return {
        'success_flag': success_flag,
        'final_outputs': final_outputs,
        'code': code,
    }

FIRST_SYSTEM_MESSAGE = '''
You are a robot exploring in an object-centric environment. Your goal is to model the logic of the world in python. You have tried it before and came up with one partially correct solution. However, it is not perfect. The code can model the logic for some experiences but failed to model the logic to achieve the goal in another environment. You need to improve your code so that the agent can achieve the objective as specified by the mission from the given initial state as well as still modelling the original logic. The new code should still follow the same template. The ` transition ` function needs to be directly runnable on (state, action) and return the next state in python. The ` reward_func ` function needs to be directly runnable on (state, action, next_state) and return (reward, done) in python.
'''

FIRST_MESSAGE = '''
Here is the partially correct solution you came up with:

```

{code}

```

However, the code failed to achieve the goal/objective as specified by the mission "{mission}" from the following initial state:

```

{initial_state}

```

The measurement for achieving the goal/objective is as follows:

```

{criterion}

```

The valid actions are {valid_actions}.

Do you know why the mission cannot be achieved from the given initial state with the world model as implemented in the code? What subgoals does the agent need to achieve in order to achieve the final goal as specified by the mission? Can the agent achieve those subgoals using the world model as implemented in the code? If not, what is missing or wrong? How can you improve the code to achieve the goal/objective as specified by the mission from the given initial state? Please improve the code as analyzed before so that the mission can be achieved from the given initial state. Please implement the code following the template. Feel free to implement any helper functions you need. You can also implement the logic for difference actions in different helper functions. However, you must implement the ` transition ` function and the ` reward_func ` function as the main functions to be called by the environment. The ` transition ` function needs to be directly runnable on (state, action) and return the next state in python. The ` reward_func ` function needs to be directly runnable on (state, action, next_state) and return (reward, done) in python. The new code, by themselves, should be complete, compilable, and runnable.
'''

CRITERION = '''
def criterion(state, mission, action, next_state, reward, done,):
    return reward > 0 and done
'''.strip()
