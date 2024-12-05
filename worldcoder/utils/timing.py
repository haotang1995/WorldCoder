#!/usr/bin/env python
# coding=utf-8

import time

# Wrapper that Just time the time used in a block
class _JustTime:
    def __init__(self):
        self.start_time = 0
        self.total_time = 0
    def __enter__(self):
        self.start()
    def __exit__(self, *args):
        self.end()
    def start(self):
        self.start_time = time.time()
    def end(self):
        self.total_time += time.time() - self.start_time
    def get(self):
        return self.total_time

class Nothing:
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass

class Timing:
    def __init__(self, enabled=True):
        self.just_timers = dict()
        self.enabled = enabled
    def __call__(self, key):
        if not self.enabled: return Nothing()
        if key not in self.just_timers:
            self.just_timers[key] = _JustTime()
        return self.just_timers[key]
    def get(self, key):
        if not self.enabled: return None
        return self.just_timers[key].get()
    def get_all(self):
        if not self.enabled: return None
        return {k: v.get() for k, v in self.just_timers.items()}
    def get_ratio(self, key):
        if not self.enabled: return None
        total_time = self.get_all()
        total_total_time = sum(total_time.values())
        return total_time[key] / total_total_time
    def get_ratio_all(self):
        if not self.enabled: return None
        total_time = self.get_all()
        total_total_time = sum(total_time.values())
        return {k: v / total_total_time for k, v in sorted(total_time.items(), key=lambda x: x[1], reverse=True)}

    def summary(self):
        if not self.enabled: return None
        total_time = self.get_all()
        return {
            'total_time': sum(total_time.values()),
            'total_time_detail': total_time,
            'total_time_ratio': self.get_ratio_all()
        }

