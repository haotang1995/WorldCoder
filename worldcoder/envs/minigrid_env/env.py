#!/usr/bin/env python
# coding=utf-8

import os
import copy
import itertools

import numpy as np
from nltk.metrics.distance import edit_distance
import gymnasium as gym
from PIL import Image

from ..base import _State, _Action, DefaultMission as Mission, ActionSpace
from .env_utils import extract_objects, AgentLocationWrapper, InventoryWrapper, GridWrapper, describe_difference, describe_objects, extract_objects2, starts_with_vowel
from .api import Agent, Key, Door, Goal, Wall, Box, Ball, Lava

class State(_State):
    def __init__(self, state, info, with_api=False, grid_repr=False):
        self.state = info['objects2']
        self.with_api = with_api
        self.grid_repr = grid_repr
    def to_hashable(self,):
        return str(tuple(sorted(self.state, key=lambda x: x['position'] if 'position' in x else (0,0))))
    def __hash__(self,):
        return hash(self.to_hashable())
    def _obj2class(self, obj, exec_globals):
        if self.with_api:
            return exec_globals[obj['type'].strip().lower().capitalize()](
                *obj['position'],
                **{
                    k: (
                        v if k not in ['carrying'] or v is None
                        else exec_globals[v['type'].strip().lower().capitalize()](
                            None, None, **{
                                kk:vv for kk,vv in v.items()
                                if kk not in ['type', 'position']
                            })
                    ) for k,v in obj.items()
                    if k not in ['type', 'position']
                }
            )
        else:
            return obj
    def to_pyrunnable(self, exec_globals=None):
        if not self.with_api:
            return self.state
        else:
            for ti in range(2):
                if exec_globals is not None and isinstance(exec_globals, dict):
                    try:
                        return {
                            self._obj2class(obj, exec_globals) for obj in self.state
                        }
                    except Exception as e:
                        if ti == 1:
                            raise e
                exec_globals = globals()
    def check_valid_pyrunnable(self, pyrunnable):
        if not self.with_api:
            if not isinstance(pyrunnable, list):
                return 'state is not a list'
            for obj in pyrunnable:
                if not isinstance(obj, dict):
                    return 'state contains non-dict object'
                if 'type' not in obj:
                    return f'object {obj} does not have a type'
                if 'position' not in obj:
                    return f'object {obj} does not have a position'
                if type(obj['position']) not in [list, tuple] or len(obj['position']) != 2:
                    return f'object {obj} does not have a valid position'
                try:
                    int(obj['position'][0])
                    int(obj['position'][1])
                except:
                    return f'object {obj} does not have a valid position with type {type(obj["position"][0])}, {type(obj["position"][1])}'
                if 'direction' in obj:
                    if type(obj['direction']) not in [list, tuple] or len(obj['direction']) != 2:
                        return f'object {obj} does not have a valid direction'
                    try:
                        int(obj['direction'][0])
                        int(obj['direction'][1])
                    except:
                        return f'object {obj} does not have a valid direction with type {type(obj["direction"][0])}, {type(obj["direction"][1])}'
                if 'color' in obj:
                    if not isinstance(obj['color'], str):
                        return f'object {obj} does not have a valid color'
                if 'state' in obj:
                    if not isinstance(obj['state'], str):
                        return f'object {obj} does not have a valid state'
                if 'carrying' in obj and obj['carrying'] is not None:
                    if not isinstance(obj['carrying'], dict):
                        return f'object {obj} does not have a valid carrying'
                    if 'type' not in obj['carrying']:
                        return f'object {obj} does not have a valid carrying since it does not have a type'
                    if 'color' in obj['carrying']:
                        if not isinstance(obj['carrying']['color'], str):
                            return f'object {obj} does not have a valid carrying since it does not have a valid color'
                    if 'state' in obj['carrying']:
                        if not isinstance(obj['carrying']['state'], str):
                            return f'object {obj} does not have a valid carrying since it does not have a valid state'
            return None
        else:
            if not hasattr(pyrunnable, '__iter__'):
                return 'state is not iterable'
            try:
                self.from_pyrunnable(pyrunnable)
            except Exception as e:
                return f'error when converting state to pyrunnable: {e}'
            return None
    def from_pyrunnable(self, pyrunnable):
        if not self.with_api:
            return self.__class__(None, {'objects2': pyrunnable}, with_api=False, grid_repr=self.grid_repr)
        else:
            objects = []
            for obj in pyrunnable:
                otype = obj.name.strip().lower()
                assert otype in ['agent', 'key', 'door', 'goal', 'wall', 'box', 'ball', 'lava'], f'object type {otype} is not supported'
                position = (int(obj.x), int(obj.y))
                o = {
                    'type': otype,
                    'position': position,
                }
                for k,v in obj.__dict__.items():
                    if k in ['x', 'y', 'name',]:
                        continue
                    if k == 'direction':
                        o[k] = (v[0], v[1])
                    elif k == 'carrying' and v is not None:
                        co = {'type': v.name.strip().lower()}
                        assert co['type'] in ['key', 'box', 'ball'], f'carrying type {co["type"]} is not supported'
                        for ck,cv in v.__dict__.items():
                            if ck not in ['x', 'y', 'name',]:
                                co[ck] = cv
                        o['carrying'] = co
                    else:
                        o[k] = v
                        if k == 'color':
                            assert v is not None, pyrunnable
                objects.append(o)
            return self.__class__(None, {'objects2': objects}, with_api=True, grid_repr=self.grid_repr)
    def to_use_in_env(self,):
        raise NotImplementedError
    def from_use_in_env(self, state, info):
        return self.__class__(state, info)
    @classmethod
    def _pos2obj(cls, state):
        output = dict()
        for obj in state:
            pos = tuple(obj['position'])
            if pos not in output:
                output[pos] = []
            sorted_obj = sorted(obj.items(), key=lambda x: x[0])
            output[pos].append(str(tuple(sorted_obj)))
        output = {k:tuple(sorted(v)) for k,v in output.items()}
        return output
    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if len(self.state) != len(other.state):
            return False
        self_pos2obj = self._pos2obj(self.state)
        other_pos2obj = self._pos2obj(other.state)
        if set(self_pos2obj.keys()) != set(other_pos2obj.keys()):
            return False
        for pos in self_pos2obj:
            if self_pos2obj[pos] != other_pos2obj[pos]:
                return False
        return True
    def __str__(self):
        if not self.grid_repr:
            pyrunnable = list(self.to_pyrunnable())
            pyrunnable = sorted(pyrunnable, key=str)
            return '{' + ', '.join([str(obj) for obj in pyrunnable]) + '}'
        else:
            max_x = max([obj['position'][0] for obj in self.state] + [1,])
            max_y = max([obj['position'][1] for obj in self.state] + [1,])
            grid = [[None for x in range(max_x+1)] for y in range(max_y+1)]
            for y in range(max_y+1):
                for x in range(max_x+1):
                    objs = [obj for obj in self.state if obj['position'] == (x,y)]
                    if len(objs) == 0:
                        grid[y][x] = 'empty' + ' ;' + ' '*5
                    else:
                        grid[y][x] = ' '.join([
                            str(self._obj2class(obj, globals()))
                            for obj in objs
                        ])
                        grid[y][x] = grid[y][x] + ' ;' + ' '*max(10-len(grid[y][x]), 0)
            return '\n'.join(['\t'.join(row) for row in grid])
    def __repr__(self):
        return self.__str__()

    @classmethod
    def _obj2str(cls, obj):
        name = obj['type']
        if 'direction' in obj:
            name = name + f" (direction={obj['direction']})"
        if 'color' in obj:
            name = obj['color'] + ' ' + name
        if 'state' in obj:
            name = obj['state'] + ' ' + name
        if 'carrying' in obj and obj['carrying'] is not None:
            name = name + f", carryinig={cls._obj2str(obj['carrying'])}"
        if 'position' in obj:
            name = (name, obj['position'][0], obj['position'][1])
        return name
    @classmethod
    def _obj_dist(cls, o1, o2):
        distance = edit_distance(o1[0], o2[0])
        distance_ratio = distance / max(len(o1[0]), len(o2[0]))
        if distance_ratio >= 0.2:
            return np.inf
        return np.linalg.norm(np.array(o1[1:]) - np.array(o2[1:])) + distance_ratio
    def diff_in_texts(self, old_state):
        if self == old_state:
            return 'Nothing happened'
        assert isinstance(old_state, State)
        old_objects = [self._obj2str(obj) for obj in old_state.state]
        new_objects = [self._obj2str(obj) for obj in self.state]
        removed_objects = [o for o in old_objects if o not in new_objects]
        added_objects = [o for o in new_objects if o not in old_objects]

        best_mapping = {}
        if min(len(removed_objects), len(added_objects)) > 0:
            matching_score = [[self._obj_dist(o1, o2) for o2 in added_objects] for o1 in removed_objects]
            matching_score = np.array(matching_score)

            # Find the best matching between removed and added objects
            matched_removed_obj_indices = np.where(np.min(matching_score, axis=1) < np.inf)[0]
            matched_added_obj_indices = np.where(np.min(matching_score, axis=0) < np.inf)[0]

            if min(len(matched_removed_obj_indices), len(matched_added_obj_indices)) > 0:
                # Assuming that there are no more than 5 objects added or removed, so that
                # we can use naive enumerate algorithm. In general, this should be a
                # Hungarian algorithm.
                assert min(len(matched_removed_obj_indices), len(matched_added_obj_indices)) <= 5
                if len(matched_added_obj_indices) >= len(matched_removed_obj_indices):
                    all_possible_permutations = list(itertools.permutations(matched_added_obj_indices))
                    best_permutation = None
                    best_permutation_score = np.inf
                    for permutation in all_possible_permutations:
                        permutation_score = sum([
                            matching_score[removed_obj_index, added_obj_index]
                            for removed_obj_index, added_obj_index in zip(matched_removed_obj_indices, permutation)
                        ])
                        if permutation_score < best_permutation_score:
                            best_permutation = permutation
                            best_permutation_score = permutation_score
                    best_mapping = dict(zip(matched_removed_obj_indices, best_permutation))
                else:
                    all_possible_permutations = list(itertools.permutations(matched_removed_obj_indices))
                    best_permutation = None
                    best_permutation_score = np.inf
                    for permutation in all_possible_permutations:
                        permutation_score = sum([
                            matching_score[removed_obj_index, added_obj_index]
                            for removed_obj_index, added_obj_index in zip(permutation, matched_added_obj_indices)
                        ])
                        if permutation_score < best_permutation_score:
                            best_permutation = permutation
                            best_permutation_score = permutation_score
                    best_mapping = dict(zip(best_permutation, matched_added_obj_indices))

        observation = ''
        for removed_obj_index, added_obj_index in best_mapping.items():
            removed_obj = removed_objects[removed_obj_index]
            added_obj = added_objects[added_obj_index]
            if removed_obj[0] == added_obj[0]:
                observation += f'The {removed_obj[0]} at pos ({removed_obj[1]}, {removed_obj[2]}) is now at pos ({added_obj[1]}, {added_obj[2]}). '
            elif str(removed_obj[1:]) == str(added_obj[1:]):
                observation += f'The {removed_obj[0]} at pos ({removed_obj[1]}, {removed_obj[2]}) becomes {"an" if starts_with_vowel(added_obj[0]) else "a"} {added_obj[0]}. '
            else:
                observation += f'The {removed_obj[0]} at pos ({removed_obj[1]}, {removed_obj[2]}) changed to {"an" if starts_with_vowel(added_obj[0]) else "a"} {added_obj[0]} at pos ({added_obj[1]}, {added_obj[2]}). '
                print(self)
                print(old_state)
                assert False, observation

        for idx in range(len(removed_objects)):
            if idx not in best_mapping:
                removed_obj = removed_objects[idx]
                observation += f'The {removed_obj[0]} at pos ({removed_obj[1]}, {removed_obj[2]}) is gone. '
        for idx in range(len(added_objects)):
            if idx not in best_mapping.values():
                added_obj = added_objects[idx]
                observation += f'There is a new {added_obj[0]} at pos ({added_obj[1]}, {added_obj[2]}). '
        return observation.strip()
    def pp(self):
        return str(self.state)
    def pp_short(self):
        state = copy.deepcopy(self.state)
        agent = next(obj for obj in state if obj['type'] == 'agent')
        objects = [
            obj
            for obj in state
            if obj['type'] != 'wall' or (
                abs(obj['position'][0]-agent['position'][0]) <= 1 and
                abs(obj['position'][1]-agent['position'][1]) <= 1
            )
        ]
        return objects
    def pp_human(self,):
        state = copy.deepcopy(self.state)
        max_x = max(obj['position'][0] for obj in state)
        max_y = max(obj['position'][1] for obj in state)
        grid = [[' ' for _ in range(max_x+1)] for _ in range(max_y+1)]
        for obj in state:
            oname = obj['type']
            if 'state' in obj:
                oname = obj['state'][0] + obj['state'][-1] + ' ' + oname
            if 'color' in obj:
                oname = obj['color'][0] + obj['color'][-1] + ' ' + oname
            if 'direction' in obj:
                oname = str(obj['direction'][0]) + str(obj['direction'][-1]) + ' ' + oname
            if 'carrying' in obj and obj['carrying']:
                oname = 'c' + obj['carrying']['type'][0] + obj['carrying']['type'][-1] + ' ' + oname
            grid[obj['position'][1]][obj['position'][0]] = oname
        return '\n'.join([', '.join([o+' '*(10-len(o)) for o in row]) for row in grid])
    def get_valid_actions(self,):
        return [Action(i) for i in range(len(ACTION_MEANINGS))]
    def get_action_space(self,):
        return [Action(i) for i in range(len(ACTION_MEANINGS))]

ACTION_MEANINGS = {
    0 : "turn left",
    1 : "turn right",
    2 : "move forward",
    3 : "pickup object",
    4 : "drop object",
    5 : "toggle",
    6 : "do nothing",
}
ACTION_MEANINGS = {
    0 : "turn left",
    1 : "turn right",
    2 : "move forward",
    3 : "pick up",
    4 : "drop",
    5 : "toggle",
    6 : "nothing",
}
MEANING_TO_ACTION = {v:k for k,v in ACTION_MEANINGS.items()}

class Action(_Action):
    def __init__(self, action):
        action = int(action)
        self.action = action
    def to_hashable(self):
        return self.action
    def to_pyrunnable(self, exec_globals=None):
        return ACTION_MEANINGS[self.action]
    @classmethod
    def check_valid_pyrunnable(cls, pyrunnable):
        return pyrunnable in ACTION_MEANINGS.values()
    @classmethod
    def from_pyrunnable(cls, pyrunnable):
        return cls(MEANING_TO_ACTION[pyrunnable])
    def to_use_in_env(self):
        return self.action
    @classmethod
    def from_use_in_env(cls, use_in_env):
        return cls(use_in_env)
    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return self.action == other.action
    def __hash__(self):
        return hash(self.action)
    def pp(self):
        return self.action

class Env:
    def __init__(self, env_name='MiniGrid-DoorKey-5x5-v0', with_api=True, grid_repr=True, seed=None,):
        self.with_api = with_api
        self.grid_repr = grid_repr
        self.name = env_name
        self.env_name = env_name
        self.detailed_carry = True
        self.dir2vec = True

        self.env = gym.make(env_name, render_mode="rgb_array")
        self.env = AgentLocationWrapper(InventoryWrapper(GridWrapper(self.env)))
        self.reset_status = 0
        self.action_space = ActionSpace([Action(i) for i in range(self.env.action_space.n)])
    def reset(self, seed=None):
        seed = seed if seed is not None else np.random.randint(1000000)
        self.state, info = self.env.reset(seed=seed)
        self.mission = self.env.mission
        grid_size1, grid_size2, _ = self.state['image'].shape
        info['grid_size'] = [grid_size1, grid_size2]
        info['objects'] = extract_objects(self.state, dir2vec=self.dir2vec)
        info['objects2'] = extract_objects2(self.state, detailed_carry=self.detailed_carry, dir2vec=self.dir2vec)
        info['mission'] = self.mission
        info['state'] = self.state
        info['state_in_text'] = f'You are in the grid of size {grid_size1}x{grid_size2}. Looking around, you see {describe_objects(extract_objects(self.state, dir2vec=self.dir2vec))}.\nYour goal is to {self.mission}.'

        output_state = State(self.state, info, with_api=self.with_api, grid_repr=self.grid_repr)
        output_mission = Mission(self.mission)
        return output_state, output_mission, info
    def step(self, action):
        action = int(action.to_use_in_env())
        new_state, reward, terminated, truncated, info = self.env.step(action)
        old_objects = extract_objects(self.state, dir2vec=self.dir2vec)
        new_objects = extract_objects(new_state, dir2vec=self.dir2vec)
        if str(sorted(old_objects)) == str(sorted(new_objects)):
            DIFFERENCE = 'Nothing happened.'
        else:
            DIFFERENCE = describe_difference(old_objects, new_objects)
        info['objects'] = new_objects
        info['objects2'] = extract_objects2(new_state, detailed_carry=self.detailed_carry, dir2vec=self.dir2vec)
        info['state'] = new_state
        info['grid_size'] = [new_state['image'].shape[0], new_state['image'].shape[1]]
        info['mission'] = self.mission
        info['different_in_text'] = DIFFERENCE
        self.state = new_state
        output_state = State(new_state, info, with_api=self.with_api, grid_repr=self.grid_repr)
        output_reward = (reward > 0)*1.
        output_done = terminated
        return output_state, output_reward, output_done, info
    @property
    def metadata(self):
        return {
            'name': self.name,
            'api': self.get_api(),
            'class': self.__class__.__name__,
        }
    def get_api(self):
        assert self.with_api, 'You need to set with_api=True when creating the environment to use this function.'
        with open(os.path.join(os.path.dirname(__file__), 'api.py')) as f:
            return f.read().strip()
    def get_success_criterion(self):
        return CRITERION
    def screenshot(self, filename):
        pixels = self.env.render()
        # use PIL
        im = Image.fromarray(pixels)
        im.save(filename)
    def get_valid_actions(self,):
        return {str(Action(i)) for i in range(self.env.action_space.n)}
CRITERION = '''
def criterion(state, mission, action, next_state, reward, done,):
    return reward > 0 and done
'''.strip()
