#!/usr/bin/env python
# coding=utf-8

class _WorldModel:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
    def predict(self, state_mission_actions):
        raise NotImplementedError
    def source_code(self,):
        raise NotImplementedError
    def save(self, path):
        raise NotImplementedError
    @classmethod
    def load(cls, path):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError
    def __eq__(self, other):
        raise NotImplementedError
    def __ne__(self, other):
        raise NotImplementedError
    def __str__(self):
        raise NotImplementedError
    def __repr__(self):
        raise NotImplementedError
