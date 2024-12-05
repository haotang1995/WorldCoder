#!/usr/bin/env python
# coding=utf-8

import copy

from .evaluator import TransitEvaluator

TRANSIT_FUNC_TEMPLATE = '''
def transition(state, action):
    """
    Args:
        state: the state of the environment
        action: the action to be executed
    Returns:
        next_state: the next state of the environment
    """
    raise NotImplementedError
'''.strip()

def get_func_template(env_metadata):
    func_template = copy.deepcopy(TRANSIT_FUNC_TEMPLATE)
    api = env_metadata['api']
    func_template = api + '\n' + func_template
    return func_template

def experiences2text(experiences):
    keys = list(experiences.keys())
    # random.shuffle(keys)
    old_state_list = [experiences[k]['state'] for k in keys]
    action_list = [experiences[k]['action'] for k in keys]
    new_state_list = [experiences[k]['state_next'] for k in keys]
    difference_list = [experiences[k]['state_next'] - experiences[k]['state'] for k in keys]
    text_experiences = [
        f'The action "{action}" transforms the state from\n```\n{old_state}\n```\nto\n```\n{new_state}\n```\nThe difference is\n"""\n{diff}\n"""\n"'.strip()
        for old_state, action, new_state, diff in zip(old_state_list, action_list, new_state_list, difference_list)
    ]
    text_experiences = '\n\n'.join(text_experiences)
    return text_experiences

def experiences2text_with_wrong_outputs(experiences, code):
    result = TransitEvaluator()(code, experiences)
    result_list = result['result_list']
    # assert all([not res['success_flag'] for res in result_list]), (f'Expect all results to be wrong, but got {result_list}', code)
    # pred_new_state_list = [res['pred_new_state'].to_pyrunnable() if res['pred_new_state'] is not None else None for res in result_list]
    pred_new_state_list = [res['pred_new_state'] if res['pred_new_state'] is not None else None for res in result_list]

    keys = list(experiences.keys())
    # random.shuffle(keys)
    old_state_list = [experiences[k]['state'] for k in keys]
    action_list = [experiences[k]['action'] for k in keys]
    new_state_list = [experiences[k]['state_next'] for k in keys]
    difference_list = [experiences[k]['state_next'] - experiences[k]['state'] for k in keys]
    compilation_error_list = [res['compilation_error'] for res in result_list]
    assert len(old_state_list) == len(new_state_list) == len(difference_list) == len(pred_new_state_list) == len(action_list), f'Expect all lists to have the same length, but got {len(old_state_list)}, {len(new_state_list)}, {len(difference_list)}, {len(pred_new_state_list)}, {len(action_list)}'

    text_experiences = [
        _exp2text_wrong(old_state, action, new_state, diff, pred_new_state, compilation_error)
        for old_state, action, new_state, diff, pred_new_state, compilation_error in zip(old_state_list, action_list, new_state_list, difference_list, pred_new_state_list, compilation_error_list)
    ]
    text_experiences = '\n\n'.join(text_experiences)
    return text_experiences
def _exp2text_wrong(old_state, action, new_state, diff, pred_new_state, compilation_error):
    if compilation_error is not None:
        return f'The action "{action}" should transform the state from\n```\n{old_state}\n```\nto\n```\n{new_state}\n```\nThe difference is\n"""\n{diff}\n"""\nHowever, the implementation fails to compile with error\n```\n{compilation_error[:compilation_error.rfind("Printed outputs:")].strip()}\n```\n'.strip()
    elif pred_new_state != new_state:
        return f'The action "{action}" should transform the state from\n```\n{old_state}\n```\nto\n```\n{new_state}\n```\nThe difference is\n"""\n{diff}\n"""\nHowever, the implementation is wrong because it returns state as \n```\n{pred_new_state}\n```\n'.strip()
    else:
        print(f'Expect the implementation to be wrong, but got correct implementation')
        return f'The action "{action}" should transform the state from\n```\n{old_state}\n```\nto\n```\n{new_state}\n```\nThe difference is\n"""\n{diff}\n"""\n'.strip()
        raise ValueError(f'Expect the implementation to be wrong, but got correct implementation')


