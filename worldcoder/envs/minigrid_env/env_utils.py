#!/usr/bin/env python
# coding=utf-8

import itertools
import numpy as np
from nltk.metrics.distance import edit_distance

from minigrid.core.constants import COLOR_NAMES, COLOR_TO_IDX, OBJECT_TO_IDX, IDX_TO_OBJECT, IDX_TO_COLOR, STATE_TO_IDX, DIR_TO_VEC
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import *  #ImgObsWrapper, RGBImgPartialObsWrapper
from gymnasium.core import ActionWrapper, ObservationWrapper, ObsType, Wrapper

DIRECTIONS2VEC = {
    i: tuple(list(vec))
    for i, vec in enumerate(DIR_TO_VEC)
}
# DIRECTIONS2VEC = {
    # 0: (1, 0),
    # 1: (0, 1),
    # 2: (-1, 0),
    # 3: (0, -1),
# }

def extract_objects(state, dir2vec=False,):

    IDX_TO_STATE = {v:k for k,v in STATE_TO_IDX.items()}

    image = state["image"]

    direction = state["direction"]
    if dir2vec:
        direction = DIRECTIONS2VEC[direction]
    agent_name = "agent (direction={}".format(direction)
    if state["carrying"] != 0:
        carried_color = IDX_TO_COLOR[state["carrying"] - 1]
        carried_type = IDX_TO_OBJECT[state["carrying_type"] - 1]
        agent_name += ", carrying=" + carried_color + " " + carried_type
    agent_name += ")"
    objects = [(agent_name, state["x-position"], state["y-position"])]

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):

            idx = image[x,y,0]
            name = IDX_TO_OBJECT[idx]
            color = IDX_TO_COLOR[image[x,y,1]]

            if name == "unseen" or name == "empty" or name == "floor":
                continue
            elif name == "door":
                # look up the state
                door_state = image[x,y,2]
                assert door_state == image[x,y,2]
                door_state = IDX_TO_STATE[door_state]
                name = f"{color} door (state={door_state})"
            elif name == "key":
                # look up the color
                name = f"{color} key"
            elif name == "goal" or name == "wall":
                pass
            elif name == 'ball':
                name = f"{color} ball"
            elif name == 'box':
                name = f"{color} box"
            elif name == 'lava':
                name = f"{color} lava"
            else:
                assert False, f"unknown object type {name}"

            objects.append((name,x,y))

    return objects

def extract_objects2(state, detailed_carry=False, dir2vec=False):
    from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR, STATE_TO_IDX

    IDX_TO_STATE = {v:k for k,v in STATE_TO_IDX.items()}

    image = state["image"]

    if state["carrying"] != 0:
        carried_color = IDX_TO_COLOR[state["carrying"] - 1]
        carried_type = IDX_TO_OBJECT[state["carrying_type"] - 1]
        if detailed_carry:
            carrying = {
                'type': carried_type,
                'color': carried_color,
            }
        else:
            carrying = carried_color + " " + carried_type
    else:
        if detailed_carry:
            carrying = None
        else:
            carrying = 'NULL'
    objects = [{
        'type': 'agent',
        'direction': state["direction"] if not dir2vec else DIRECTIONS2VEC[state["direction"]],
        'carrying': carrying,
        'position': (state["x-position"], state["y-position"]),
    }]

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):

            idx = image[x,y,0]
            name = IDX_TO_OBJECT[idx]
            color = IDX_TO_COLOR[image[x,y,1]]

            if name == "unseen" or name == "empty" or name == "floor":
                continue
            elif name == "door":
                # look up the state
                door_state = image[x,y,2]
                assert door_state == image[x,y,2]
                door_state = IDX_TO_STATE[door_state]
                obj = {
                    'type': 'door',
                    'color': color,
                    'state': door_state,
                    'position': (x,y),
                }
            elif name == "key":
                # look up the color
                obj = {
                    'type': 'key',
                    'color': color,
                    'position': (x,y),
                }
            elif name == "goal" or name == "wall":
                obj = {
                    'type': name,
                    'position': (x,y),
                }
            elif name == 'ball':
                ball_state = image[x,y,2]
                assert ball_state == image[x,y,2]
                ball_state = IDX_TO_STATE[ball_state]
                obj = {
                    'type': 'ball',
                    'color': color,
                    # 'state': ball_state,
                    'position': (x,y),
                }
            elif name == 'box':
                box_state = image[x,y,2]
                assert box_state == image[x,y,2]
                box_state = IDX_TO_STATE[box_state]
                obj = {
                    'type': 'box',
                    'color': color,
                    # 'state': box_state,
                    'position': (x,y),
                }
            elif name == 'lava':
                lava_state = image[x,y,2]
                assert lava_state == image[x,y,2]
                lava_state = IDX_TO_STATE[lava_state]
                obj = {
                    'type': 'lava',
                    'color': color,
                    # 'state': lava_state,
                    'position': (x,y),
                }
            else:
                assert False, f"unknown object type {name}"

            objects.append(obj)

    return objects

class InventoryWrapper(ObservationWrapper):
    """
    Adds the `.carrying` field to the agent's observations.
    """

    def __init__(self, env):
        super().__init__(env)

        # one carrying option for each color

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "carrying": spaces.Discrete(1+len(COLOR_NAMES)), "carrying_type": spaces.Discrete(1+len(OBJECT_TO_IDX))}
        )


    def observation(self, obs):
        env = self.unwrapped
        carrying = env.carrying
        if carrying is None:
            carrying_color = 0
            carrying_type = 0
        else:
            carrying_color = COLOR_TO_IDX[carrying.color] + 1
            carrying_type = OBJECT_TO_IDX[carrying.type] + 1

        return {**obs, "carrying": carrying_color, "carrying_type": carrying_type}

class AgentLocationWrapper(ObservationWrapper):
    """
    Adds the agent location and direction
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces,
            "x-position": spaces.Discrete(self.env.width),
            "y-position": spaces.Discrete(self.env.height),
            "direction": spaces.Discrete(4)}
        )

    def observation(self, obs):
        env = self.unwrapped

        x,y = env.agent_pos
        direction = env.agent_dir
        return {**obs, "x-position": x, "y-position": y, "direction": direction}

class GridWrapper(ObservationWrapper):
    """shows the grid objects"""

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces,
            "image": spaces.Box(low=0, high=255, shape=(self.env.width, self.env.height, 3), dtype="uint8")}
        )

    def observation(self, obs):
        env = self.unwrapped

        image = env.grid.encode()
        return {**obs, "image": image}

def starts_with_vowel(word):
    return word[0] in 'aeiou'
def describe_objects(objects):
    return ', '.join([f'{"an" if starts_with_vowel(ot) else "a"} {ot} at pos ({x}, {y})' for ot, x, y in sorted(objects)])

def match(o1, o2):
    distance = edit_distance(o1[0], o2[0])
    distance_ratio = distance / max(len(o1[0]), len(o2[0]))
    if distance_ratio >= 0.2:
        return np.inf
    return np.linalg.norm(np.array(o1[1:]) - np.array(o2[1:])) + distance_ratio
def describe_difference(old_objects, new_objects):
    removed_objects = [o for o in old_objects if o not in new_objects]
    added_objects = [o for o in new_objects if o not in old_objects]

    best_mapping = {}
    if min(len(removed_objects), len(added_objects)) > 0:
        matching_score = [[match(o1, o2) for o2 in added_objects] for o1 in removed_objects]
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
