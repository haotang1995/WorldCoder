import numpy as np
import copy

from .synthesizer import synthesis
from .guess_reward import guess_reward as _guess_reward
from .llm_utils import LLM

def add_synthesis_args(parser):
    parser.add_argument('--strategy', type=str, default='bandits')
    parser.add_argument('--no_plan_obj_flag', dest='plan_obj_flag', action='store_false', default=True)
    parser.add_argument('--no_sep_plan_flag', dest='sep_plan_flag', action='store_false', default=True)
    parser.add_argument('--llm_max_step_num', type=int, default=None)
    parser.add_argument('--synthesis_budget', type=int, default=1000000)
    parser.add_argument('--bandits_c', type=float, default=5.0)
    parser.add_argument('--no_verbose', dest='verbose', action='store_false')
    parser.add_argument('--llm_model_name', type=str, default='gpt-4')
    parser.add_argument('--llm_temperature', type=float, default=1.0)
    return parser
def get_synthesis_args(args):
    if args.llm_max_step_num is None:
        if 'env' not in args:
            args.max_step_num = 50
        elif 'minigrid' in args.env.lower():
            args.max_step_num = 50
        elif 'sokoban' in args.env.lower():
            args.max_step_num = 30
        else:
            args.max_step_num = 50
    else:
        args.max_step_num = args.llm_max_step_num

    return {
        'strategy': args.strategy,
        'plan_obj_flag': args.plan_obj_flag,
        'sep_plan_flag': args.sep_plan_flag,
        'max_step_num': args.max_step_num,
        'budget': args.synthesis_budget,
        'bandits_C': args.bandits_c,
        'llm_default_args': {
            'model': args.llm_model_name,
            'temperature': args.llm_temperature,
        },
        'verbose': args.verbose,
    }
def refine_world_model(
    init_transit_code,
    init_reward_code,
    experiences,
    envs_to_plan,
    key_missions,
    env_metadata=None,
    llm_default_args = {'model': 'gpt-4', 'temperature': 1.0,},
    strategy='bandits',
    verbose=False,
    max_step_num=300, budget=1000000,
    plan_obj_flag=True,
    sep_plan_flag=True,
    bandits_C=5.0,
    np_rng=None,
):
    # assert strategy in ['bandits', 'greedy', 'random']
    assert strategy in ['bandits',]
    assert len(experiences) > 0
    assert len(envs_to_plan) >= 0
    assert isinstance(np_rng, np.random.Generator)
    print(f'refine_world_model with {len(experiences)} experiences and {len(envs_to_plan)} envs_to_plan')

    llm = LLM(seed=0, default_args=llm_default_args)

    whole_reward_code = copy.deepcopy(init_reward_code)
    best_output, cache_dir = synthesis(
        experiences,
        envs_to_plan,
        key_missions,
        init_transit_code=init_transit_code,
        init_reward_code=init_reward_code,
        strategy=strategy,
        env_metadata=env_metadata,
        llm=llm,
        np_rng=np_rng,
        verbose=verbose,
        max_step_num=max_step_num, budget=budget,
        plan_obj_flag=plan_obj_flag and not sep_plan_flag,
        bandits_C=bandits_C,
    )
    transit_code = best_output['transit_code']
    reward_code = best_output['reward_code']
    if sep_plan_flag and plan_obj_flag:
        best_output2, cache_dir2 = synthesis(
            experiences,
            envs_to_plan,
            key_missions,
            init_transit_code=transit_code,
            init_reward_code=reward_code,
            env_metadata=env_metadata,
            llm=llm,
            strategy=strategy,
            np_rng=np_rng,
            verbose=verbose,
            max_step_num=max_step_num, budget=budget,
            plan_obj_flag=True,
            bandits_C=bandits_C,
        )
        transit_code = best_output2['transit_code']
        reward_code = best_output2['reward_code']
        if 'total_costs' not in best_output2 and 'total_costs' in best_output:
            best_output2['total_costs'] = best_output['total_costs']
        elif 'total_costs' in best_output and 'total_costs' in best_output2:
            for k in best_output2['total_costs']:
                best_output2['total_costs'][k] += best_output['total_costs'][k]
        if 'total_new_costs' not in best_output2 and 'total_new_costs' in best_output:
            best_output2['total_new_costs'] = best_output['total_new_costs']
        elif 'total_new_costs' in best_output and 'total_new_costs' in best_output2:
            for k in best_output2['total_new_costs']:
                best_output2['total_new_costs'][k] += best_output['total_new_costs'][k]
        best_output2['no_plan_output'] = best_output
        best_output = best_output2
    assert all([str(m) in reward_code for m in key_missions])
    if whole_reward_code is not None:
        whole_reward_code.update(reward_code)
    else:
        whole_reward_code = reward_code
    fitness = best_output['success_ratio']
    return fitness, (transit_code, whole_reward_code, best_output, cache_dir)

def guess_reward(reward_code, mission, llm_default_args, **kwargs):
    llm = LLM(default_args=llm_default_args)
    output = _guess_reward(reward_code, mission, llm,)
    _code = output['code']
    output['total_costs'] = {k: [v] for k, v in output['final_outputs']['costs'].items()}
    output['total_new_costs'] = {k: [v] for k, v in output['final_outputs']['new_costs'].items()}
    return _code, output


