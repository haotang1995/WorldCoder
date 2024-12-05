#!/usr/bin/env python
# coding=utf-8

from .base import _WorldModel

class OracleWorldModel(_WorldModel):
    def __init__(self, env,):
        self.env = env
    def register(self, seed, state, mission):
        self.mission = mission
        self.env_cache = {state: [seed,]}
        self.cache = dict()
    def predict(self, state_mission_actions):
        predictions = []
        for state, mission, action in state_mission_actions:
            if (state, mission, action) in self.cache:
                predictions.append(self.cache[(state, mission, action)])
            else:
                assert mission == self.mission
                path = self.env_cache[state]
                env = self.env
                env.reset(path[0])
                for a in path[1:]:
                    env.step(a)
                next_state, reward, done, info = env.step(action)
                predictions.append((next_state, reward, done,))
                self.cache[(state, mission, action)] = (next_state, reward, done)
                self.env_cache[next_state] = path + [action]
        return predictions

