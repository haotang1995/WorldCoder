#!/usr/bin/env python
# coding=utf-8

import copy
import hashlib
import numpy as np

from ...utils.caching import _CacheSystem
from ...utils.eval_code import eval_code
from ...envs.base import _State, _Action

from ..world_model import WorldModel
from ..planner import planner

# class _Evaluator(_CacheSystem):
    # def __init__(self, seed=0,):
        # super(_Evaluator, self).__init__(seed=seed, stochastic=False)
class _Evaluator:
    def __init__(self, seed=0,):
        pass
    def __call__(self, *args, **kwargs):
        return self._action(*args, **kwargs)

class TransitEvaluator(_Evaluator):
    def __init__(self, seed=0,):
        super(TransitEvaluator, self).__init__(seed=seed)
    def _cache_id(self, code, experiences):
        assert isinstance(code, str)
        code_id = str(len(code)) + '_' + hashlib.md5(code.encode()).hexdigest()
        exp_id = str(len(experiences)) + '_' + hashlib.md5(str(sorted(experiences, key=str,)).encode()).hexdigest()
        return (
            ('code', code_id, code),
            ('experiences', exp_id, experiences),
        )
    def _action(self, code, experiences):
        results = self.__class__._evaluate_transit_code(code, experiences)
        return results

    @staticmethod
    def _get_transit_func(code):
        compilation_error, func_name, transit_func = None, None, None
        exec_globals = {}
        exec_globals = eval_code(code, exec_globals=exec_globals, return_exec_globals=True)
        if not isinstance(exec_globals, dict):
            compilation_error = exec_globals
            return compilation_error, func_name, transit_func, exec_globals

        if 'transition' in exec_globals and callable(exec_globals['transition']):
            func_name = {'transition': exec_globals['transition']}
        else:
            func_name = {k:v for k, v in exec_globals.items() if not k.startswith('_') and callable(v)}
            if len(func_name) == 0:
                compilation_error = 'No transition function found'
                return compilation_error, func_name, transit_func, exec_globals
            elif len(func_name) > 1:
                tmp_func_name = {k:v for k, v in func_name.items() if 'transit' in k}
                if len(tmp_func_name) >= 1:
                    func_name = tmp_func_name
            if len(func_name) > 1:
                print(f'Warning: Expect only one transition function, but got {len(func_name)}')
                lastest_k = list(func_name.keys())[-1]
                func_name = {lastest_k: func_name[lastest_k]}
        func_name = list(func_name.keys())[0]
        transit_func = exec_globals[func_name]
        assert callable(transit_func), f'Expect {func_name} to be callable, but got {transit_func}'
        return compilation_error, func_name, transit_func, exec_globals
    @staticmethod
    def _evaluate_transit_code(code, experiences):
        compilation_error, func_name, transit_func, exec_globals = TransitEvaluator._get_transit_func(code)
        if compilation_error is not None:
            results = {
                'success_flag': False,
                'success_ratio': 0,
                'compilation_error': compilation_error,
                'crt_experiences': dict(),
                'wrong_experiences': experiences,
                'experiences': experiences,
                'result_list': [{
                    'success_flag': False,
                    'pred_new_state': None,
                    'pred_state_success_flag': False,
                    'experience': exp,
                    'compilation_error': compilation_error,
                } for exp in experiences.values()],
                'func_name': None,
            }
            return results

        result_list = [TransitEvaluator._eval_transit_per_experience(transit_func, exp, exec_globals,) for exp in experiences.values()]
        success_flag = all([result['success_flag'] for result in result_list])
        success_ratio = sum([result['success_flag'] for result in result_list]) / len(result_list)
        pred_state_success_flag = all([result['pred_state_success_flag'] for result in result_list])
        pred_state_success_ratio = sum([result['pred_state_success_flag'] for result in result_list]) / len(result_list)
        crt_experiences = {k: v for (k, v), result in zip(experiences.items(), result_list) if result['success_flag']}
        wrong_experiences = {k: v for (k, v), result in zip(experiences.items(), result_list) if not result['success_flag']}
        results = {
            'success_flag': success_flag,
            'success_ratio': success_ratio,
            'pred_state_success_flag': pred_state_success_flag,
            'pred_state_success_ratio': pred_state_success_ratio,
            'compilation_error': None,
            'crt_experiences': crt_experiences,
            'wrong_experiences': wrong_experiences,
            'experiences': experiences,
            'result_list': result_list,
            'func_name': func_name,
        }
        return results
    @staticmethod
    def _eval_transit_per_experience(transit_func, experience, exec_globals):
        assert isinstance(experience, dict), f'Expect experience to be a dict, but got {experience}'
        assert 'state' in experience, f'Expect experience to have key "state", but got {list(experience.keys())}'
        assert 'state_next' in experience, f'Expect experience to have key "state_next", but got {list(experience.keys())}'
        assert isinstance(experience['state'], _State), f'Expect experience["state"] to be an instance of _State, but got {experience["state"]}'
        assert isinstance(experience['state_next'], _State), f'Expect experience["state_next"] to be an instance of _State, but got {experience["state_next"]}'
        assert 'action' in experience, f'Expect experience to have key "action", but got {list(experience.keys())}'
        assert isinstance(experience['action'], _Action), f'Expect experience["action"] to be an instance of _Action, but got {experience["action"]}'

        code_to_run = "old_state = experience['state'].to_pyrunnable(exec_globals=exec_globals)"
        _exec_globals = eval_code(code_to_run, exec_globals=locals(), return_exec_globals=True)
        if isinstance(_exec_globals, str):
            pred_new_state = None
            pred_state_success_flag = False
            return {
                'success_flag': False,
                'pred_state_success_flag': pred_state_success_flag,
                'pred_new_state': pred_new_state,
                'experience': experience,
                'compilation_error': _exec_globals,
            }
        old_state = _exec_globals['old_state']
        code_to_run = "action = experience['action'].to_pyrunnable(exec_globals=exec_globals)"
        _exec_globals = eval_code(code_to_run, exec_globals=locals(), return_exec_globals=True)
        if isinstance(_exec_globals, str):
            pred_new_state = None
            pred_state_success_flag = False
            return {
                'success_flag': False,
                'pred_state_success_flag': pred_state_success_flag,
                'pred_new_state': pred_new_state,
                'experience': experience,
                'compilation_error': _exec_globals,
            }
        action = _exec_globals['action']
        new_state = experience['state_next']

        copied_old_state = copy.deepcopy(old_state)
        copied_action = copy.deepcopy(action)
        exec_globals.update({
            'copied_old_state': copied_old_state,
            'copied_action': copied_action,
            'transit_func': transit_func,
        })
        code_to_exec = 'pred_new_state = transit_func(copied_old_state, copied_action)'
        exec_globals = eval_code(code_to_exec, exec_globals=exec_globals, return_exec_globals=True)
        if isinstance(exec_globals, str):
            pred_new_state = None
            pred_state_success_flag = False
            success_flag = False
            compilation_error = exec_globals
        else:
            compilation_error = None
            pred_new_state = exec_globals['pred_new_state']
            valid_state = new_state.check_valid_pyrunnable(pred_new_state)
            if isinstance(valid_state, str):
                pred_new_state = None
                pred_state_success_flag = False
                compilation_error = f'The predicted new state is invalid with the following error: {valid_state.strip()}'
            else:
                pred_new_state = new_state.from_pyrunnable(pred_new_state)
                pred_state_success_flag = (new_state == pred_new_state)
            success_flag = pred_state_success_flag
        result = {
            'success_flag': success_flag,
            'pred_new_state': pred_new_state,
            'pred_state_success_flag': pred_state_success_flag,
            'experience': experience,
            'compilation_error': compilation_error,
        }
        return result

class RewardEvaluator(_Evaluator):
    def __init__(self, seed=0,):
        super(RewardEvaluator, self).__init__(seed=seed)
    def _cache_id(self, code, experiences):
        assert isinstance(code, str)
        code_id = str(len(code)) + '_' + hashlib.md5(code.encode()).hexdigest()
        exp_id = str(len(experiences)) + '_' + hashlib.md5(str(sorted(experiences, key=str,)).encode()).hexdigest()
        return (
            ('code', code_id, code),
            ('experiences', exp_id, experiences),
        )
    def _action(self, code, experiences):
        results = self.__class__._evaluate_reward_code(code, experiences)
        return results
    @staticmethod
    def _get_reward_func(code):
        compilation_error, func_name, reward_func = None, None, None
        exec_globals = {}
        exec_globals = eval_code(code, exec_globals=exec_globals, return_exec_globals=True)
        if not isinstance(exec_globals, dict):
            compilation_error = exec_globals
            return compilation_error, func_name, reward_func, exec_globals

        if 'reward_func' in exec_globals:
            func_name = {'reward_func': exec_globals['reward_func']}
        else:
            func_name = {k:v for k, v in exec_globals.items() if not k.startswith('_') and callable(v)}
            if len(func_name) == 0:
                compilation_error = 'No transition function found'
                return compilation_error, func_name, reward_func, exec_globals
            elif len(func_name) > 1:
                tmp_func_name = {k:v for k, v in func_name.items() if 'reward' in k}
                if len(tmp_func_name) >= 1:
                    func_name = tmp_func_name
            if len(func_name) > 1:
                print(f'Warning: Expect only one reward function, but got {len(func_name)}')
                lastest_k = list(func_name.keys())[-1]
                func_name = {lastest_k: func_name[lastest_k]}
        func_name = list(func_name.keys())[0]
        reward_func = exec_globals[func_name]
        assert callable(reward_func), f'Expect {func_name} to be callable, but got {reward_func}'
        return compilation_error, func_name, reward_func, exec_globals
    @staticmethod
    def _evaluate_reward_code(code, experiences):
        compilation_error, func_name, reward_func, exec_globals = RewardEvaluator._get_reward_func(code)
        if compilation_error is not None:
            results = {
                'success_flag': False,
                'success_ratio': 0,
                'compilation_error': compilation_error,
                'crt_experiences': dict(),
                'wrong_experiences': experiences,
                'experiences': experiences,
                'result_list': [{
                    'success_flag': False,
                    'pred_reward_success_flag': False,
                    'pred_done_success_flag': False,
                    'pred_reward': None,
                    'pred_done': None,
                    'experience': exp,
                    'compilation_error': compilation_error,
                } for exp in experiences.values()],
                'func_name': None,
            }
            return results

        result_list = [RewardEvaluator._eval_reward_per_experience(reward_func, exp, exec_globals,) for exp in experiences.values()]
        success_flag = all([result['success_flag'] for result in result_list])
        success_ratio = sum([result['success_flag'] for result in result_list]) / len(result_list)
        pred_reward_success_flag = all([result['pred_reward_success_flag'] for result in result_list])
        pred_reward_success_ratio = sum([result['pred_reward_success_flag'] for result in result_list]) / len(result_list)
        pred_done_success_flag = all([result['pred_done_success_flag'] for result in result_list])
        pred_done_success_ratio = sum([result['pred_done_success_flag'] for result in result_list]) / len(result_list)
        crt_experiences = {k: v for (k, v), result in zip(experiences.items(), result_list) if result['success_flag']}
        wrong_experiences = {k: v for (k, v), result in zip(experiences.items(), result_list) if not result['success_flag']}
        results = {
            'success_flag': success_flag,
            'success_ratio': success_ratio,
            'pred_reward_success_flag': pred_reward_success_flag,
            'pred_reward_success_ratio': pred_reward_success_ratio,
            'pred_done_success_flag': pred_done_success_flag,
            'pred_done_success_ratio': pred_done_success_ratio,
            'compilation_error': None,
            'crt_experiences': crt_experiences,
            'wrong_experiences': wrong_experiences,
            'experiences': experiences,
            'result_list': result_list,
            'func_name': func_name,
        }
        return results
    def _eval_reward_per_experience(reward_func, experience, exec_globals):
        assert isinstance(experience, dict), f'Expect experience to be a dict, but got {experience}'
        assert 'state' in experience, f'Expect experience to have key "state", but got {list(experience.keys())}'
        assert 'state_next' in experience, f'Expect experience to have key "state_next", but got {list(experience.keys())}'
        assert isinstance(experience['state'], _State), f'Expect experience["state"] to be an instance of _State, but got {experience["state"]}'
        assert isinstance(experience['state_next'], _State), f'Expect experience["state_next"] to be an instance of _State, but got {experience["state_next"]}'
        assert 'action' in experience, f'Expect experience to have key "action", but got {list(experience.keys())}'
        assert isinstance(experience['action'], _Action), f'Expect experience["action"] to be an instance of _Action, but got {experience["action"]}'
        assert 'reward' in experience, f'Expect experience to have key "reward", but got {list(experience.keys())}'
        assert 'done' in experience, f'Expect experience to have key "done", but got {list(experience.keys())}'

        code_to_run = "old_state = experience['state'].to_pyrunnable(exec_globals=exec_globals)"
        _exec_globals = eval_code(code_to_run, exec_globals=locals(), return_exec_globals=True)
        if isinstance(_exec_globals, str):
            pred_reward = None
            pred_done = None
            pred_reward_success_flag = False
            pred_done_success_flag = False
            return {
                'success_flag': False,
                'pred_reward_success_flag': False,
                'pred_done_success_flag': False,
                'pred_reward': pred_reward,
                'pred_done': pred_done,
            }
        old_state = _exec_globals['old_state']

        code_to_run = "new_state = experience['state_next'].to_pyrunnable(exec_globals=exec_globals)"
        _exec_globals = eval_code(code_to_run, exec_globals=locals(), return_exec_globals=True)
        if isinstance(_exec_globals, str):
            pred_reward = None
            pred_done = None
            pred_reward_success_flag = False
            pred_done_success_flag = False
            return {
                'success_flag': False,
                'pred_reward_success_flag': False,
                'pred_done_success_flag': False,
                'pred_reward': pred_reward,
                'pred_done': pred_done,
            }
        new_state = _exec_globals['new_state']

        code_to_run = "action = experience['action'].to_pyrunnable(exec_globals=exec_globals)"
        _exec_globals = eval_code(code_to_run, exec_globals=locals(), return_exec_globals=True)
        if isinstance(_exec_globals, str):
            pred_reward = None
            pred_done = None
            pred_reward_success_flag = False
            pred_done_success_flag = False
            return {
                'success_flag': False,
                'pred_reward_success_flag': False,
                'pred_done_success_flag': False,
                'pred_reward': pred_reward,
                'pred_done': pred_done,
            }
        action = _exec_globals['action']
        reward = experience['reward']
        done = experience['done']

        copied_old_state = copy.deepcopy(old_state)
        copied_action = copy.deepcopy(action)
        copied_new_state = copy.deepcopy(new_state)
        exec_globals.update({
            'copied_old_state': copied_old_state,
            'copied_action': copied_action,
            'copied_new_state': copied_new_state,
            'reward_func': reward_func,
        })
        code_to_exec = 'pred_reward, pred_done = reward_func(copied_old_state, copied_action, copied_new_state)'
        exec_globals = eval_code(code_to_exec, exec_globals=exec_globals, return_exec_globals=True)
        if isinstance(exec_globals, str):
            pred_reward = None
            pred_done = None
            pred_reward_success_flag = False
            pred_done_success_flag = False
            success_flag = False
            compilation_error = exec_globals
        else:
            compilation_error = None
            pred_reward = exec_globals['pred_reward']
            try:
                pred_reward_success_flag = abs(reward - pred_reward) < 1e-6
            except Exception as e:
                pred_reward_success_flag = False
                compilation_error = f'Failed to compare the reward with the following error: {e}'
            pred_done = exec_globals['pred_done']
            pred_done_success_flag = (done == pred_done)
            success_flag = pred_reward_success_flag and pred_done_success_flag
        result = {
            'success_flag': success_flag,
            'pred_reward_success_flag': pred_reward_success_flag,
            'pred_done_success_flag': pred_done_success_flag,
            'pred_reward': pred_reward,
            'pred_done': pred_done,
            'experience': experience,
            'compilation_error': compilation_error,
        }
        return result

class PlanEvaluator(_Evaluator):
    def __init__(self, seed=0,):
        super(PlanEvaluator, self).__init__(seed=seed)
    def _cache_id(self, transit_code, reward_code, envs, key_missions):
        assert isinstance(transit_code, str), f'Expect code to be a string, but got {transit_code}'
        assert isinstance(reward_code, dict), f'Expect code to be a dict, but got {reward_code}'
        key_missions = list(sorted(set([str(mission) for mission in key_missions])))
        reward_code = [reward_code[m] for m in key_missions]
        transit_code_id = str(len(transit_code)) + '_' + hashlib.md5(transit_code.encode()).hexdigest()
        reward_code_id = str(len(reward_code)) + '_' + hashlib.md5(str(reward_code).encode()).hexdigest()
        envs = {k: {kk:vv for kk, vv in v.items() if kk not in ['info']} for k, v in envs.items()}
        envs_id = str(len(envs)) + '_' + self._value2id(envs)
        key_missions_id = str(len(key_missions)) + '_' + hashlib.md5(str(sorted(key_missions)).encode()).hexdigest()
        return (
            ('transit_code', transit_code_id, transit_code),
            ('reward_code', reward_code_id, reward_code),
            ('envs', envs_id, envs),
            ('key_missions', key_missions_id, key_missions),
        )
    def _action(self, transit_code, reward_code, envs, key_missions):
        results = self.__class__._evaluate_code_for_plan(transit_code, reward_code, envs, key_missions)
        return results
    @staticmethod
    def _evaluate_code_for_plan(transit_code, reward_code, envs, key_missions):
        envs = SepPlanEvaluator._filter_w_missions(envs, key_missions,)
        assert isinstance(envs, dict), f'Expect envs to be a dict, but got {envs}'
        if len(envs) == 0:
            return {
                'success_flag': True,
                'success_ratio': 1.,
                'compilation_error': None,
                'crt_envs': dict(),
                'wrong_envs': envs,
                'envs': envs,
                'result_list': None,
            }

        result_list = [
            SepPlanEvaluator()(copy.deepcopy(transit_code), copy.deepcopy(reward_code), copy.deepcopy(envs[env_id]))
            for env_id in envs
        ]

        success_flag = all([res['success_flag'] for res in result_list])
        success_ratio = sum([res['success_ratio'] for res in result_list]) / len(result_list)
        compilation_error = None
        crt_envs = {k: v for (k, v), result in zip(envs.items(), result_list) if result['success_flag']}
        wrong_envs = {k: v for (k, v), result in zip(envs.items(), result_list) if not result['success_flag']}
        return {
            'success_flag': success_flag,
            'success_ratio': success_ratio,
            'compilation_error': compilation_error,
            'crt_envs': crt_envs,
            'wrong_envs': wrong_envs,
            'envs': envs,
            'key_missions': key_missions,
            'result_list': result_list,
        }
class SepPlanEvaluator(_Evaluator):
    def __init__(self, seed=0,):
        super(SepPlanEvaluator, self).__init__(seed=seed)
    def _cache_id(self, transit_code, reward_code, env):
        assert isinstance(transit_code, str), f'Expect code to be a string, but got {transit_code}'
        if isinstance(reward_code, str):
            _reward_code = reward_code
            reward_code = dict()
            reward_code[str(env['mission'])] = _reward_code
        assert isinstance(reward_code, dict), f'Expect code to be a dict, but got {reward_code}'
        reward_code = reward_code[str(env['mission'])]
        assert isinstance(reward_code, str), f'Expect code to be a string, but got {reward_code}'
        assert isinstance(env, dict), f'Expect env to be a dict, but got {env}'
        transit_code_id = str(len(transit_code)) + '_' + hashlib.md5(transit_code.encode()).hexdigest()
        reward_code_id = str(len(reward_code)) + '_' + hashlib.md5(reward_code.encode()).hexdigest()
        env_id = hashlib.md5(str(sorted(env.items(), key=str)).encode()).hexdigest()
        return (
            ('transit_code', transit_code_id, transit_code),
            ('reward_code', reward_code_id, reward_code),
            ('env', env_id, env),
        )
    def _action(self, transit_code, reward_code, env):
        results = self.__class__._evaluate_code_for_plan_per_env_with_code(transit_code, reward_code, env)
        return results

    @staticmethod
    def _evaluate_code_for_plan_per_env_with_code(transit_code, reward_code, env,):
        assert isinstance(transit_code, str), f'Expect code to be a string, but got {transit_code}'
        if isinstance(reward_code, str):
            _reward_code = reward_code
            reward_code = dict()
            reward_code[str(env['mission'])] = _reward_code
        assert isinstance(reward_code, dict), f'Expect code to be a dict, but got {reward_code}'
        assert str(env['mission']) in reward_code, f'Expect {env["mission"]} to be in reward_code, but got {reward_code}'
        assert isinstance(reward_code[str(env['mission'])], str), f'Expect reward_code[{env["mission"]}] to be a string, but got {reward_code[str(env["mission"])]}'
        world_model = WorldModel(transit_code, reward_code)
        if str(env['mission']) in world_model.exec_vars and \
                world_model.exec_vars[str(env['mission'])]['compile_error'] is not None:
            return {
                'success_flag': False,
                'success_ratio': 0.,
                'compilation_error': world_model.exec_vars[str(env['mission'])]['compile_error'],
                'env': env,
                'trajectories': None,
            }
        return SepPlanEvaluator._evaluate_code_for_plan_per_env(world_model, env)
    @staticmethod
    def _evaluate_code_for_plan_per_env(world_model, env,):
        state = env['state']
        mission = env['mission']
        planning_options = env['planning_options']

        crt, compilation_error = False, None
        trajectories = []

        plan = planner(
            initial_state=state, mission=mission,
            world_model=world_model,
            **planning_options
        )
        if plan is None:
            compilation_error = 'No successors found for state {}'.format(state)
            return {
                'success_flag': False,
                'success_ratio': 0.,
                'compilation_error': compilation_error,
                'env': env,
                'trajectories': trajectories,
            }
        for ai, action in enumerate(plan):
            predictions = world_model.predict([(state, mission, action)])[0]
            if isinstance(predictions, Exception):
                # assert False, f'Expect predictions to be a tuple of (new_state, reward, done), but got {predictions}'
                compilation_error = str(predictions)
                return {
                    'success_flag': False,
                    'success_ratio': 0.,
                    'compilation_error': compilation_error,
                    'env': env,
                    'trajectories': trajectories,
                }
            new_state, reward, done = predictions

            trajectories.append((state, mission, action, new_state, reward, done))
            crt = reward > 0 and done
            if crt: break;
            if done: break;
            state = new_state
        success_flag = crt if crt is not None else False
        return {
            'success_flag': success_flag,
            'success_ratio': float(success_flag),
            'compilation_error': compilation_error,
            'env': env,
            'trajectories': trajectories,
        }
    @staticmethod
    def _filter_w_missions(data, missions):
        missions = set([str(mission) for mission in missions])
        assert isinstance(data, dict)
        return {k:v for k, v in data.items() if str(v['mission']) in missions}

class JointEvaluator(_Evaluator):
    def __init__(self, seed=0,):
        super(JointEvaluator, self).__init__(seed=seed)

    def _cache_id(self, transit_code, reward_code, experiences, envs_to_plan, key_missions, plan_obj_flag=False,):
        assert isinstance(transit_code, str), f'Expect transit_code to be str, but got {transit_code}'
        assert isinstance(reward_code, dict), f'Expect reward_code to be dict, but got {reward_code}'
        key_missions = list(sorted(set([str(mission) for mission in key_missions])))
        # reward_code = [reward_code[m] for m in key_missions]
        # experiences = {k: v for k, v in experiences.items() if str(v['mission']) in key_missions}
        # envs_to_plan = {k: v for k, v in envs_to_plan.items() if str(v['mission']) in key_missions}
        transit_code_id = str(len(transit_code)) + '_' + hashlib.md5(transit_code.encode()).hexdigest()
        reward_code_id = str(len(reward_code)) + '_' + self._value2id(reward_code)
        exp_id = str(len(experiences)) + '_' + hashlib.md5(str(sorted(experiences, key=str)).encode()).hexdigest()
        envs_to_plan = {k: {kk:vv for kk, vv in v.items() if kk not in ['info']} for k, v in envs_to_plan.items()}
        env_id = str(len(envs_to_plan)) + '_' + self._value2id(envs_to_plan)
        key_missions_id = str(len(key_missions)) + '_' + hashlib.md5(str(sorted(key_missions)).encode()).hexdigest()
        plan_obj_flag_id = hashlib.md5(str(plan_obj_flag).encode()).hexdigest()
        return (
            ('transit_code', transit_code_id, transit_code),
            ('reward_code', reward_code_id, reward_code),
            ('experiences', exp_id, experiences),
            ('envs_to_plan', env_id, envs_to_plan),
            ('key_missions', key_missions_id, key_missions),
            ('plan_obj_flag', plan_obj_flag_id, plan_obj_flag),
        )
    def _action(self, transit_code, reward_code, experiences, envs_to_plan, key_missions, plan_obj_flag=False,):
        results = self.__class__._joint_evaluate(transit_code, reward_code, experiences, envs_to_plan, key_missions, plan_obj_flag,)
        return results

    @staticmethod
    def _joint_evaluate(
        transit_code, reward_code, experiences, envs_to_plan, key_missions, plan_obj_flag=False,
    ):
        print('Evaluating code using experiences...')
        exp_result = JointEvaluator._joint_evaluate_code(transit_code, reward_code, experiences, key_missions,)
        if plan_obj_flag:
            exp_in_key_missions = SepPlanEvaluator._filter_w_missions(experiences, key_missions)
            env_in_key_missions = SepPlanEvaluator._filter_w_missions(envs_to_plan, key_missions)
            data_ratio = len(exp_in_key_missions) / (len(exp_in_key_missions) + len(env_in_key_missions))
            if not exp_result['success_flag']:
                plan_result = {
                    'success_flag': None,
                    'success_ratio': None,
                }
                success_flag = exp_result['success_flag']
                success_ratio = exp_result['success_ratio'] * data_ratio
            else:
                print('Evaluating code using planning envs...')
                plan_result = PlanEvaluator()(transit_code, reward_code, envs_to_plan, key_missions,)
                success_flag = exp_result['success_flag'] and plan_result['success_flag']
                success_ratio = data_ratio * exp_result['success_ratio'] + (1-data_ratio) * plan_result['success_ratio']
        else:
            data_ratio = 1
            plan_result = {
                'success_flag': None,
                'success_ratio': None,
            }
            success_flag = exp_result['success_flag']
            success_ratio = exp_result['success_ratio'] * data_ratio
        return {
            'transit_code': transit_code,
            'reward_code': reward_code,
            'success_flag': success_flag,
            'success_ratio': success_ratio,
            'exp_result': exp_result,
            'plan_result': plan_result,
            'configurations':{
                'data_ratio': data_ratio,
                'plan_obj_flag': plan_obj_flag,
                'per_datasize_norm_flag': True,
                'experiences': experiences,
                'envs_to_plan': envs_to_plan,
                'key_missions': key_missions,
            }
        }
    @staticmethod
    def _joint_evaluate_code_per_mission(transit_code, reward_code, experiences,):
        assert isinstance(transit_code, str), f'Expect transit_code to be str, but got {transit_code}'
        assert isinstance(reward_code, str), f'Expect reward_code to be str, but got {reward_code}'
        sep_transit_result = TransitEvaluator()(transit_code, experiences,)
        sep_reward_result = RewardEvaluator()(reward_code, experiences,)

        # compilation_error, transit_func_name, transit_func, reward_func_name, reward_func, exec_globals = _get_funcs(transit_code, reward_code)
        # if compilation_error is not None:
            # results = {
                # 'success_flag': False,
                # 'success_ratio': 0,
                # 'transit_success_flag': transit_func is not None,
                # 'reward_success_flag': False,
                # 'transit_success_ratio': 0 if transit_func is None else 1,
                # 'reward_success_ratio': 0,
                # 'compilation_error': compilation_error,
                # 'crt_experiences': dict(),
                # 'wrong_experiences': experiences,
                # 'crt_transit_experiences': dict(),
                # 'crt_reward_experiences': dict(),
                # 'wrong_transit_experiences': experiences if transit_func is None else dict(),
                # 'wrong_reward_experiences': experiences,
                # 'experiences': experiences,
                # 'result_list': [{'success_flag': False,} for _ in range(len(experiences))],
                # 'transit_func_name': transit_func_name,
                # 'reward_func_name': reward_func_name,
            # }
            # return results
        # transit_result_list = [eval_transit_per_experience(transit_func, exp, exec_globals,) for exp in experiences.values()]
        # reward_result_list = [eval_reward_per_experience(reward_func, exp, exec_globals,) for exp in experiences.values()]
        result_list = [
            {
                'success_flag': transit_result['success_flag'] and reward_result['success_flag'],
                'success_ratio': (transit_result['pred_state_success_flag'] + reward_result['pred_reward_success_flag'] + reward_result['pred_done_success_flag']) / 3.,
                'transit_success_flag': transit_result['success_flag'],
                'reward_success_flag': reward_result['success_flag'],
                'pred_new_state': transit_result['pred_new_state'],
                'pred_reward': reward_result['pred_reward'],
                'pred_done': reward_result['pred_done'],
                'pred_state_success_flag': transit_result['pred_state_success_flag'],
                'pred_reward_success_flag': reward_result['pred_reward_success_flag'],
                'pred_done_success_flag': reward_result['pred_done_success_flag'],
                'compilation_error': transit_result['compilation_error'] or reward_result['compilation_error'],
                'transit_compilation_error': transit_result['compilation_error'],
                'reward_compilation_error': reward_result['compilation_error'],
                'experience': exp,
            }
            # for exp, transit_result, reward_result in zip(experiences.values(), transit_result_list, reward_result_list)
            for exp, transit_result, reward_result in zip(experiences.values(), sep_transit_result['result_list'], sep_reward_result['result_list'])
        ]
        # for res, tres, rres in zip(result_list, sep_transit_result_list, sep_reward_result_list):
            # assert res['pred_state_success_flag'] == tres['pred_state_success_flag'], (res, tres, transit_code, reward_code)
            # assert res['pred_reward_success_flag'] == rres['pred_reward_success_flag'], (res, rres, transit_code, reward_code)
            # assert res['pred_done_success_flag'] == rres['pred_done_success_flag'], (res, rres, transit_code, reward_code)

        success_flag = all([result['success_flag'] for result in result_list])
        success_ratio = float(np.mean([result['success_ratio'] for result in result_list]))
        transit_success_flag = all([result['pred_state_success_flag'] for result in result_list])
        transit_success_ratio = sum([result['pred_state_success_flag'] for result in result_list]) / len(result_list)
        reward_success_flag = all([result['pred_reward_success_flag'] and result['pred_done_success_flag'] for result in result_list])
        reward_success_ratio = sum([result['pred_reward_success_flag'] and result['pred_done_success_flag'] for result in result_list]) / len(result_list)
        pred_state_success_flag = all([result['pred_state_success_flag'] for result in result_list])
        pred_state_success_ratio = sum([result['pred_state_success_flag'] for result in result_list]) / len(result_list)
        pred_reward_success_flag = all([result['pred_reward_success_flag'] for result in result_list])
        pred_reward_success_ratio = sum([result['pred_reward_success_flag'] for result in result_list]) / len(result_list)
        pred_done_success_flag = all([result['pred_done_success_flag'] for result in result_list])
        pred_done_success_ratio = sum([result['pred_done_success_flag'] for result in result_list]) / len(result_list)
        crt_transit_experiences = {k: v for (k, v), result in zip(experiences.items(), result_list) if result['pred_state_success_flag']}
        crt_reward_experiences = {k: v for (k, v), result in zip(experiences.items(), result_list) if result['pred_reward_success_flag'] and result['pred_done_success_flag']}
        crt_experiences = {k: v for (k, v), result in zip(experiences.items(), result_list) if result['success_flag']}
        wrong_transit_experiences = {k: v for (k, v), result in zip(experiences.items(), result_list) if not result['pred_state_success_flag']}
        wrong_reward_experiences = {k: v for (k, v), result in zip(experiences.items(), result_list) if not result['pred_reward_success_flag'] or not result['pred_done_success_flag']}
        wrong_experiences = {k: v for (k, v), result in zip(experiences.items(), result_list) if not result['success_flag']}
        assert transit_success_flag or len(wrong_transit_experiences)
        assert reward_success_flag or len(wrong_reward_experiences)
        assert success_flag or len(wrong_experiences)
        results = {
            'success_flag': success_flag,
            'success_ratio': success_ratio,
            'transit_success_flag': transit_success_flag,
            'transit_success_ratio': transit_success_ratio,
            'reward_success_flag': reward_success_flag,
            'reward_success_ratio': reward_success_ratio,
            'pred_state_success_flag': pred_state_success_flag,
            'pred_state_success_ratio': pred_state_success_ratio,
            'pred_reward_success_flag': pred_reward_success_flag,
            'pred_reward_success_ratio': pred_reward_success_ratio,
            'pred_done_success_flag': pred_done_success_flag,
            'pred_done_success_ratio': pred_done_success_ratio,
            'compilation_error': None,
            'crt_experiences': crt_experiences,
            'wrong_experiences': wrong_experiences,
            'crt_transit_experiences': crt_transit_experiences,
            'crt_reward_experiences': crt_reward_experiences,
            'wrong_transit_experiences': wrong_transit_experiences,
            'wrong_reward_experiences': wrong_reward_experiences,
            'experiences': experiences,
            'result_list': result_list,
            'transit_func_name': sep_transit_result['func_name'],
            'reward_func_name': sep_reward_result['func_name'],
        }
        return results
    @staticmethod
    def _joint_evaluate_code(transit_code, reward_code, experiences, key_missions,):
        experiences = SepPlanEvaluator._filter_w_missions(experiences, key_missions)
        key_missions = set([str(mission) for mission in key_missions])
        assert isinstance(transit_code, str), f'Expect transit_code to be str, but got {transit_code}'
        assert isinstance(reward_code, dict), f'Expect reward_code to be dict, but got {reward_code}'

        result_list = [None for _ in range(len(experiences))]
        list_experiences = list(experiences.items())
        for mission, rc in reward_code.items():
            if mission not in key_missions:
                continue
            exp_idx = [idx for idx, (_, exp) in enumerate(list_experiences) if str(exp['mission']) == str(mission)]
            if len(exp_idx) == 0:
                continue
            cur_experiences = {k: v for k, v in list_experiences if str(v['mission']) == str(mission)}
            cur_res = JointEvaluator._joint_evaluate_code_per_mission(transit_code, rc, cur_experiences)
            assert len(exp_idx) == len(cur_res['result_list']), (exp_idx, cur_res['result_list'])
            for idx, res in zip(exp_idx, cur_res['result_list']):
                result_list[idx] = res
        iid_pred_reward_success_ratio = sum([result['pred_reward_success_flag'] for result in result_list if result is not None]) / len([result for result in result_list if result is not None])
        iid_pred_done_success_ratio = sum([result['pred_done_success_flag'] for result in result_list if result is not None]) / len([result for result in result_list if result is not None])

        ood_missions = {str(exp['mission']) for exp, res in zip(experiences.values(), result_list) if res is None}.intersection(key_missions)
        assert not ood_missions, ood_missions

        non_key_missions = {str(exp['mission']) for exp, res in zip(experiences.values(), result_list) if res is None}.difference(key_missions)
        assert not non_key_missions, non_key_missions
        exp_idx = [idx for idx, (_, exp) in enumerate(list_experiences) if str(exp['mission']) in non_key_missions]
        non_key_experiences = {k: v for k, v in experiences.items() if str(v['mission']) in non_key_missions}
        if non_key_missions:
            cur_result_list = JointEvaluator._joint_evaluate_code_per_mission(transit_code, '', non_key_experiences)['result_list']
            assert len(exp_idx) == len(cur_result_list), (exp_idx, cur_result_list)
            for idx, res in zip(exp_idx, cur_result_list):
                assert res['pred_state_success_flag'] == res['transit_success_flag'], (res['pred_state_success_flag'], res['transit_success_flag'])
                res = {
                    'success_flag': res['transit_success_flag'],
                    'success_ratio': bool(res['transit_success_flag'])*1.,
                    'transit_success_flag': res['transit_success_flag'],
                    'reward_success_flag': None,
                    'pred_new_state': res['pred_new_state'],
                    'pred_reward': None,
                    'pred_done': None,
                    'pred_state_success_flag': res['pred_state_success_flag'],
                    'pred_reward_success_flag': None,
                    'pred_done_success_flag': None,
                    'compilation_error': res['transit_compilation_error'],
                    'transit_compilation_error': res['transit_compilation_error'],
                    'reward_compilation_error': None,
                    'experience': res['experience'],
                }
                result_list[idx] = res


        assert len(result_list) == len(experiences), (len(result_list), len(experiences))
        assert all([res is not None for res in result_list])

        success_flag = all([result['success_flag'] for result in result_list])
        success_ratio = float(np.mean([result['success_ratio'] for result in result_list]))
        transit_success_flag = all([result['pred_state_success_flag'] for result in result_list])
        transit_success_ratio = sum([result['pred_state_success_flag'] for result in result_list]) / len(result_list)
        reward_success_flag = all([result['pred_reward_success_flag'] and result['pred_done_success_flag'] for result in result_list if str(result['experience']['mission']) in key_missions])
        reward_success_ratio = float(np.mean([result['pred_reward_success_flag'] and result['pred_done_success_flag'] for result in result_list if str(result['experience']['mission']) in key_missions]))
        pred_state_success_flag = all([result['pred_state_success_flag'] for result in result_list])
        pred_state_success_ratio = sum([result['pred_state_success_flag'] for result in result_list]) / len(result_list)
        pred_reward_success_flag = all([result['pred_reward_success_flag'] for result in result_list if str(result['experience']['mission']) in key_missions])
        pred_reward_success_ratio = float(np.mean([result['pred_reward_success_flag'] for result in result_list if str(result['experience']['mission']) in key_missions]))
        pred_done_success_flag = all([result['pred_done_success_flag'] for result in result_list if str(result['experience']['mission']) in key_missions])
        pred_done_success_ratio = float(np.mean([result['pred_done_success_flag'] for result in result_list if str(result['experience']['mission']) in key_missions]))
        crt_transit_experiences = {k: v for (k, v), result in zip(experiences.items(), result_list) if result['pred_state_success_flag']}
        crt_reward_experiences = {k: v for (k, v), result in zip(experiences.items(), result_list) if result['pred_reward_success_flag'] and result['pred_done_success_flag'] and str(result['experience']['mission']) in key_missions}
        crt_experiences = {k: v for (k, v), result in zip(experiences.items(), result_list) if result['success_flag']}
        all_wrong_transit_experiences = {k: v for (k, v), result in zip(experiences.items(), result_list) if not result['pred_state_success_flag']}
        all_wrong_reward_experiences = {k: v for (k, v), result in zip(experiences.items(), result_list) if not (result['pred_reward_success_flag'] and result['pred_done_success_flag']) and str(result['experience']['mission']) in key_missions}
        all_wrong_experiences = {k: v for (k, v), result in zip(experiences.items(), result_list) if not result['success_flag']}
        wrong_transit_experiences = all_wrong_transit_experiences
        wrong_experiences = all_wrong_experiences
        wrong_reward_experiences = all_wrong_reward_experiences

        assert abs(success_ratio - (pred_reward_success_ratio + pred_done_success_ratio + pred_state_success_ratio) / 3.) < 1e-6, (success_ratio, pred_reward_success_ratio, pred_done_success_ratio, pred_state_success_ratio)

        results = {
            'success_flag': success_flag,
            'success_ratio': success_ratio,
            'transit_success_flag': transit_success_flag,
            'transit_success_ratio': transit_success_ratio,
            'reward_success_flag': reward_success_flag,
            'reward_success_ratio': reward_success_ratio,
            'pred_state_success_flag': pred_state_success_flag,
            'pred_state_success_ratio': pred_state_success_ratio,
            'pred_reward_success_flag': pred_reward_success_flag,
            'pred_reward_success_ratio': pred_reward_success_ratio,
            'pred_done_success_flag': pred_done_success_flag,
            'pred_done_success_ratio': pred_done_success_ratio,
            'compilation_error': None,
            'crt_experiences': crt_experiences,
            'wrong_experiences': wrong_experiences,
            'all_wrong_experiences': all_wrong_experiences,
            'crt_transit_experiences': crt_transit_experiences,
            'crt_reward_experiences': crt_reward_experiences,
            'wrong_transit_experiences': wrong_transit_experiences,
            'wrong_reward_experiences': wrong_reward_experiences,
            'all_wrong_transit_experiences': all_wrong_transit_experiences,
            'all_wrong_reward_experiences': all_wrong_reward_experiences,
            'experiences': experiences,
            'key_missions': key_missions,
            'result_list': result_list,
        }
        return results
