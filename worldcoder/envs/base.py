#!/usr/bin/env python
# coding=utf-8

class Base:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
    def to_hashable(self):
        raise NotImplementedError
    def to_pyrunnable(self):
        raise NotImplementedError
    @classmethod
    def check_valid_pyrunnable(cls, pyrunnable):
        raise NotImplementedError
    @classmethod
    def from_pyrunnable(cls, pyrunnable):
        raise NotImplementedError
    def to_use_in_env(self):
        raise NotImplementedError
    @classmethod
    def from_use_in_env(cls, use_in_env):
        raise NotImplementedError
    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.to_hashable() == other.to_hashable()
    def __ne__(self, other):
        return not self == other
    def __hash__(self):
        return hash(self.to_hashable())
    def __str__(self):
        return str(self.to_pyrunnable())
    def __repr__(self):
        return repr(self.to_pyrunnable())
    def pp(self):
        raise NotImplementedError
    def __lt__(self, other):
        return str(self.to_hashable()) < str(other.to_hashable())

class _State(Base):
    def __sub__(self, old_state):
        return self.diff_in_texts(old_state)
    def diff_in_texts(self, old_state):
        raise NotImplementedError
    def valid_actions(self):
        raise NotImplementedError

class _Mission(Base):
    pass

class _Action(Base):
    pass

class DefaultMission(_Mission):
    def __init__(self, mission='win the game'):
        mission = str(mission)
        self.mission = mission
    def to_hashable(self):
        return self.mission
    def to_pyrunnable(self):
        return self.mission
    @classmethod
    def check_valid_pyrunnable(cls, pyrunnable):
        return isinstance(pyrunnable, str)
    @classmethod
    def from_pyrunnable(cls, pyrunnable):
        return cls(pyrunnable)
    def to_use_in_env(self):
        return self.mission
    @classmethod
    def from_use_in_env(cls, use_in_env='win the game'):
        return cls(use_in_env)
    def __eq__(self, other):
        return self.mission == other.mission
    def __hash__(self):
        return hash(self.mission)
    def pp(self):
        return self.mission

class ActionSpace:
    def __init__(self, action_space):
        self.action_space = tuple(action_space)
        self.n = len(self.action_space)
    def sample(self):
        return random.choice(self.action_space)
    def contains(self, action):
        return action in self.action_space
    def __eq__(self, other):
        return self.action_space == other.action_space
    def __hash__(self):
        return hash(self.action_space)
    def __str__(self):
        return str(self.action_space)
    def __repr__(self):
        return repr(self.action_space)
    def __len__(self):
        return self.n
    def __iter__(self):
        return iter(self.action_space)
    def __getitem__(self, i):
        return self.action_space[i]
    def __setitem__(self, i, action):
        raise NotImplementedError('ActionSpace is immutable')
        self.action_space[i] = action
    def __contains__(self, action):
        return action in self.action_space
