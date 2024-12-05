import os
import dill
from collections import defaultdict
import copy
import json
import difflib

from ...utils.eval_code import eval_code, swallow_io

class WorldModel:
    def __init__(self, transit_code, reward_code, synthesizer_logs=None,):
        assert isinstance(transit_code, str), f'Expect transit_code to be a string, but got {transit_code}'
        assert isinstance(reward_code, dict), f'Expect reward_code to be a dict, but got {reward_code}'
        self.transit_code = transit_code
        self.reward_code = reward_code
        self.synthesizer_logs = synthesizer_logs
        self.exec_vars = dict()
        for mission, rc in self.reward_code.items():
            compile_error, transit_func_name, transit_func, reward_func_name, reward_func, exec_globals = _get_funcs(self.transit_code, rc)
            self.exec_vars[mission] = {
                'compile_error': compile_error,
                'transit_func_name': transit_func_name,
                'transit_func': transit_func,
                'reward_func_name': reward_func_name,
                'reward_func': reward_func,
                'exec_globals': exec_globals,
            }

    def __hash__(self):
        return hash((self.transit_code, self.reward_code))
    def __eq__(self, other):
        return isinstance(other, WorldModel) and self.transit_code.strip() == other.transit_code.strip() and self.reward_code == other.reward_code
    def __ne__(self, other):
        return not self.__eq__(other)

    def predict(self, state_mission_actions):
        predictions = []
        for state, mission, action in state_mission_actions:
            mission = str(mission)
            assert mission in self.reward_code, f'Mission {mission} not found in reward_code'
            if mission not in self.exec_vars:
                rc = self.reward_code[mission]
                compile_error, transit_func_name, transit_func, reward_func_name, reward_func, exec_globals = _get_funcs(self.transit_code, rc)
                self.exec_vars[mission] = {
                    'compile_error': compile_error,
                    'transit_func_name': transit_func_name,
                    'transit_func': transit_func,
                    'reward_func_name': reward_func_name,
                    'reward_func': reward_func,
                    'exec_globals': exec_globals,
                }
            compile_error = self.exec_vars[mission]['compile_error']
            transit_func_name = self.exec_vars[mission]['transit_func_name']
            transit_func = self.exec_vars[mission]['transit_func']
            reward_func_name = self.exec_vars[mission]['reward_func_name']
            reward_func = self.exec_vars[mission]['reward_func']
            exec_globals = self.exec_vars[mission]['exec_globals']
            if compile_error is not None:
                predictions.append(Exception(compile_error[:compile_error.rfind('Printed output:')].strip()))
                continue

            state = copy.deepcopy(state)
            action = copy.deepcopy(action)

            old_state = state.to_pyrunnable(exec_globals)

            error_message = None
            try:
                with swallow_io() as s:
                    old_action = action.to_pyrunnable(exec_globals)
                try:
                    with swallow_io() as s:
                        new_state = transit_func(copy.deepcopy(old_state), copy.deepcopy(old_action))
                    new_state = state.from_pyrunnable(new_state).to_pyrunnable(exec_globals)
                    try:
                        with swallow_io() as s:
                            new_reward, new_done = reward_func(copy.deepcopy(old_state), copy.deepcopy(old_action), copy.deepcopy(new_state))
                        try:
                            new_reward = float(new_reward)
                            try:
                                new_done = bool(new_done)
                            except Exception as e:
                                error_message = e
                                # raise e
                        except Exception as e:
                            error_message = e
                            # raise e
                    except Exception as e:
                        new_reward = None
                        new_done = None
                        error_message = e
                        # raise e
                except Exception as e:
                    new_state = None
                    new_reward = None
                    new_done = None
                    error_message = e
                    # raise e
            except Exception as e:
                old_action = None
                new_state = None
                new_reward = None
                new_done = None
                error_message = e
                # raise e

            if error_message is not None:
                predictions.append(error_message)
                continue
            if new_state is None or new_reward is None or new_done is None:
                error_message = 'new_state, new_reward, new_done cannot be None but got {}, {}, {}'.format(new_state, new_reward, new_done)
                predictions.append(Exception(error_message))
                continue

            # make sure everything has the right type
            error_message = state.check_valid_pyrunnable(new_state)
            if error_message is not None:
                predictions.append(Exception(error_message))
                continue
            new_state = state.from_pyrunnable(new_state)
            predictions.append((new_state, new_reward, new_done))

        return predictions

    def source_code(self):
        return {
            'transit': self.transit_code,
            'reward': self.reward_code,
        }

    def save(self, path):
        with open(os.path.join(path, 'transit.py'), "w") as f:
            f.write(self.transit_code)
        with open(os.path.join(path, 'reward.json'), "w") as f:
            json.dump(self.reward_code, f)
        with open(os.path.join(path, 'synthesizer_logs.dill'), "wb") as f:
            dill.dump(self.synthesizer_logs, f)
    @classmethod
    def load(cls, path):
        with open(os.path.join(path, 'transit.py'), "r") as f:
            transit_code = f.read()
        with open(os.path.join(path, 'reward.json'), "r") as f:
            reward_code = json.load(f)
        with open(os.path.join(path, 'synthesizer_logs.dill'), "rb") as f:
            synthesizer_logs = dill.load(f)
        return cls(transit_code, reward_code, synthesizer_logs)

def _extract_api(transit_code, reward_code):
    api = ''
    while True:
        transit_lines = transit_code.split('\n')
        reward_lines = reward_code.split('\n')
        if len(transit_lines) == 0 or len(reward_lines) == 0:
            break
        match = difflib.SequenceMatcher(None, transit_lines, reward_lines).find_longest_match(0, len(transit_lines), 0, len(reward_lines))
        match_start_index, match_end_index = match.a, match.a + match.size
        while match_start_index < match_end_index and (
            len(transit_lines[match_start_index]) == 0 or
                len(transit_lines[match_start_index]) != len(transit_lines[match_start_index].lstrip())):
            match_start_index += 1
        while match_start_index < match_end_index and (
            len(transit_lines[match_end_index-1]) == 0 or
            (match_end_index+1 < len(transit_lines) and len(transit_lines[match_end_index]) != len(transit_lines[match_end_index].lstrip()))):
            match_end_index -= 1
        if match_start_index == match_end_index:
            break
        api_candidate = '\n'.join(transit_lines[match_start_index:match_end_index])
        exec_globals = eval_code(api+'\n'+api_candidate, exec_globals={}, return_exec_globals=True)
        if not isinstance(exec_globals, dict):
            break
        api += '\n' + api_candidate
        assert transit_code.count(api_candidate) == 1, (f'Expect transit_code to contain api_candidate only once, but got {transit_code.count(api_candidate)}', api_candidate, transit_code)
        transit_code = transit_code.replace(api_candidate, '\n')
        assert reward_code.count(api_candidate) == 1, (f'Expect reward_code to contain api_candidate only once, but got {reward_code.count(api_candidate)}', api_candidate, reward_code)
        reward_code = reward_code.replace(api_candidate, '\n')
    return api, transit_code, reward_code

def _extract_api_deprecated(transit_code, reward_code):
    lines = transit_code.split('\n')
    api = ''
    first_index = -1
    for index, line in enumerate(lines):
        if len(line.strip()) == 0:
            continue
        if len(line.lstrip()) != len(line):
            continue
        if line.startswith('class'):
            if first_index == -1:
                first_index = index
        elif first_index != -1:
            last_index = index
            break
    if first_index != -1:
        api = '\n'.join(lines[first_index:last_index])
    else:
        api = ''
    return api
def _get_funcs(transit_code, reward_code):
    # Check if transit_code and reward_code are valid
    assert isinstance(transit_code, str), f'Expect codes to be a string, but got {transit_code}'
    assert isinstance(reward_code, str), f'Expect codes to be a string, but got {reward_code}'
    exec_globals = eval_code(transit_code, exec_globals={}, return_exec_globals=True)
    if not isinstance(exec_globals, dict):
        compilation_error = exec_globals
        return compilation_error, None, None, None, None, exec_globals
    exec_globals = eval_code(reward_code, exec_globals={}, return_exec_globals=True)
    if not isinstance(exec_globals, dict):
        compilation_error = exec_globals
        return compilation_error, None, None, None, None, exec_globals

    _original_transit_code, _original_reward_code = transit_code, reward_code
    api, transit_code, reward_code = _extract_api(transit_code, reward_code)

    compilation_error, transit_func_name, transit_func, reward_func_name, reward_func = None, None, None, None, None
    common_exec_globals = {}
    common_exec_globals = eval_code(api, exec_globals=common_exec_globals, return_exec_globals=True)
    if not isinstance(common_exec_globals, dict):
        print(api)
        assert False, (f'Expect common_exec_globals to be dict, but got {common_exec_globals}', api, transit_code, reward_code)
        compilation_error = common_exec_globals
        return compilation_error, transit_func_name, transit_func, reward_func_name, reward_func, common_exec_globals

    print(f'non-copyable: {[k for k, v in common_exec_globals.items() if not copyable(v)]}')
    common_exec_globals = {k:v for k, v in common_exec_globals.items() if copyable(v)}
    transit_exec_globals = eval_code(transit_code, exec_globals=copy.deepcopy(common_exec_globals), return_exec_globals=True)
    if not isinstance(transit_exec_globals, dict):
        assert False, (f'Expect transit_exec_globals to be dict, but got {transit_exec_globals}', api, transit_code, reward_code, _original_transit_code, _original_reward_code)
        compilation_error = transit_exec_globals
        return compilation_error, transit_func_name, transit_func, reward_func_name, reward_func, transit_exec_globals
    if 'transition' in transit_exec_globals:
        transit_func_name = {'transition': transit_exec_globals['transition']}
    else:
        transit_func_name = {k:v for k, v in transit_exec_globals.items() if not k.startswith('_') and callable(v)}
        if len(transit_func_name) == 0:
            compilation_error = 'No transition function found'
            return compilation_error, transit_func_name, transit_func, reward_func_name, reward_func, transit_exec_globals
        elif len(transit_func_name) > 1:
            tmp_transit_func_name = {k:v for k, v in transit_func_name.items() if 'transit' in k}
            if len(tmp_transit_func_name) >= 1:
                transit_func_name = tmp_transit_func_name
        if len(transit_func_name) > 1:
            print(f'Warning: Expect only one transition function, but got {len(transit_func_name)}')
            lastest_k = list(transit_func_name.keys())[-1]
            transit_func_name = {lastest_k: transit_func_name[lastest_k]}
    transit_func_name = list(transit_func_name.keys())[0]
    transit_func = transit_exec_globals[transit_func_name]
    assert callable(transit_func), f'Expect {transit_func_name} to be callable, but got {transit_func}'

    reward_exec_globals = eval_code(reward_code, exec_globals=copy.deepcopy(common_exec_globals), return_exec_globals=True)
    if not isinstance(reward_exec_globals, dict):
        assert False, (f'Expect reward_exec_globals to be dict, but got {reward_exec_globals}', api, transit_code, reward_code)
        compilation_error = reward_exec_globals
        return compilation_error, transit_func_name, transit_func, reward_func_name, reward_func, reward_exec_globals
    if 'reward_func' in reward_exec_globals:
        reward_func_name = {'reward_func': reward_exec_globals['reward_func']}
    else:
        reward_func_name = {k:v for k, v in reward_exec_globals.items() if not k.startswith('_') and callable(v)}
        if len(reward_func_name) == 0:
            compilation_error = 'No transition function found'
            return compilation_error, transit_func_name, transit_func, reward_func_name, reward_func, reward_exec_globals
        elif len(reward_func_name) > 1:
            tmp_reward_func_name = {k:v for k, v in reward_func_name.items() if 'reward' in k}
            if len(tmp_reward_func_name) >= 1:
                reward_func_name = tmp_reward_func_name
        if len(reward_func_name) > 1:
            print(f'Warning: Expect only one reward function, but got {len(reward_func_name)}')
            lastest_k = list(reward_func_name.keys())[-1]
            reward_func_name = {lastest_k: reward_func_name[lastest_k]}
    reward_func_name = list(reward_func_name.keys())[0]
    reward_func = reward_exec_globals[reward_func_name]
    assert callable(reward_func), f'Expect {reward_func_name} to be callable, but got {reward_func}'

    common_exec_globals.update(transit_exec_globals)
    common_exec_globals.update(reward_exec_globals)
    return compilation_error, transit_func_name, transit_func, reward_func_name, reward_func, common_exec_globals

def copyable(data):
    try:
        copy.deepcopy(data)
        return True
    except:
        # print(f'Not copyable: {data}')
        return False
