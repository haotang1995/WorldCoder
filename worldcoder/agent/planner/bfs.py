#!/usr/bin/env python
# coding=utf-8

import multiprocessing as mp

import time
import hashlib
from pprint import pprint

from ...utils.timing import Timing

def prepare_candidates(inp):
    state, path = inp
    return [
        (state, action, path+(action,)) for action in state.get_valid_actions()
    ]

class predict:
    def __init__(self, world_model, mission):
        self.world_model = world_model
        self.mission = mission
    def __call__(self, inp):
        state, action, path = inp
        return self.world_model.predict([(state, self.mission, action)])[0], path

def breadth_first_search(
    state, mission, world_model,
    budget=100, max_depth=30,
):
    print(f'state len: {len(state.to_pyrunnable())}')
    timer = Timing(enabled=True)
    pool = mp.Pool(processes=4)
    visited = {state,}
    frontier = [(state, (),),]
    depth = 0
    start_time = time.time()
    while depth < max_depth:
        new_frontier = []
        with timer('prepare_candidates'):
            candidates = pool.map(
                prepare_candidates,
                frontier,
            )
        print(f'average candidates: {sum(len(c) for c in candidates)/len(candidates)}')
        with timer('flat_candidates'):
            candidates = [item for sublist in candidates for item in sublist]
        with timer('predict'):
            for pred, path in pool.imap_unordered(
                predict(world_model, mission),
                candidates,
            ):
                with timer('process_pred'):
                    if isinstance(pred, (list, tuple)) and len(pred) == 3:
                        new_state, reward, done = pred
                        if done and reward > 0:
                            return path
                        if new_state in visited:
                            continue
                        visited.add(new_state)
                        new_frontier.append((new_state, path,))
                    else:
                        continue
        with timer('logging & printing'):
            frontier = new_frontier
            depth += 1
            if time.time() - start_time > budget:
                break
            print(f'depth: {depth}, frontier: {len(frontier)}, candidates: {len(candidates)}, visited: {len(visited)}, time: {time.time()-start_time}')
        pprint(world_model.timer.summary())
        print('-'*80)
        pprint(timer.summary())

    paths = list(sorted([p for s, p in frontier], key=str))
    index = hashlib.md5(str(paths).encode()).digest()[0] % len(paths)
    return paths[index]
