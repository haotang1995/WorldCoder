#!/usr/bin/env python
# coding=utf-8

import os
import dill
try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, create_engine, select, ARRAY, Float
from sqlalchemy.orm import Session

# from .bfs import breadth_first_search
from .mcts import mcts

Base = declarative_base()
class PlanCache(Base):  # type: ignore
    """SQLite table for MCTS Cache (all generations)."""
    __tablename__ = "plan_cache"
    initial_state= Column(String, primary_key=True)
    mission = Column(String, primary_key=True)
    world_model = Column(String, primary_key=True)
    method = Column(String, primary_key=True)
    budget = Column(Integer, primary_key=True)
    max_depth = Column(Integer, primary_key=True)
    ucb_c = Column(Float, primary_key=True)
    idx = Column(Integer,)
ENGINE = create_engine('sqlite:///' + os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plan_cache.db'))

class PlannerWithSuccess:
    def __init__(self,):
        self.cache_schema = PlanCache
        self.cache_schema.metadata.create_all(ENGINE)
    def __call__(self, initial_state, mission, world_model, method='mcts', budget=100, max_depth=30, ucb_c=1.0):
        initial_state_in_str = str(initial_state)
        mission_in_str = str(mission)
        transit_code, reward_code = world_model.source_code()['transit'], world_model.source_code()['reward']
        world_model_in_str = transit_code + '\n\n' + '\n'.join([k+':\n'+reward_code[k] for k in sorted(reward_code.keys())])

        # Retrieve the cache
        stmt = (
            select(
                self.cache_schema.idx,
            ).where(
                self.cache_schema.initial_state == initial_state_in_str,
                self.cache_schema.mission == mission_in_str,
                self.cache_schema.world_model == world_model_in_str,
                self.cache_schema.method == method,
                self.cache_schema.budget == budget,
                self.cache_schema.max_depth == max_depth,
                self.cache_schema.ucb_c == ucb_c,
            ).order_by(self.cache_schema.idx)
        )
        with Session(ENGINE) as session:
            generations = session.execute(stmt).fetchall()
            generations = [row for row in generations]
            generations.sort(key=lambda x: x[-1])
            assert len(generations) <= 1, f"generations: {generations}"
            if len(generations) == 1:
                return _fetch_cache(generations[0][0])

        # Actual planning
        result = self._action(initial_state, mission, world_model, method=method, budget=budget, max_depth=max_depth, ucb_c=ucb_c)

        # Get max idx in cache
        stmt = (
            select(
                self.cache_schema.idx,
            ).order_by(self.cache_schema.idx)
        )
        with Session(ENGINE) as session, session.begin():
            generations = session.execute(stmt).fetchall()
            generations = [row for row in generations]
            generations.sort(key=lambda x: x[-1])
            max_idx = generations[-1][0] if generations else -1
            assert max_idx == len(generations) - 1, f"max_idx: {max_idx}, generations: {len(generations)}"
            item = self.cache_schema(
                initial_state=initial_state_in_str,
                mission=mission_in_str,
                world_model=world_model_in_str,
                method=method,
                budget=budget,
                max_depth=max_depth,
                ucb_c=ucb_c,
                idx=max_idx+1,
            )
            session.add(item)
        _store_cache(max_idx+1, result)
        return result

    def _action(
        self, initial_state, mission, world_model,
        method='bfs', budget=30000, max_depth=30, ucb_c=1.0,
    ):
        if method == 'bfs':
            raise NotImplementedError
            plan = breadth_first_search(
                initial_state, mission, world_model,
                budget=budget, max_depth=max_depth,
            )
            success = plan is not None
        elif method == 'mcts':
            plan, success = mcts(
                initial_state, mission, world_model,
                budget=budget, max_depth=max_depth, ucb_c=ucb_c,
            )
        else:
            raise NotImplementedError
        return plan, success

PLAN_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plan_cache')
os.makedirs(PLAN_CACHE_DIR, exist_ok=True)
def _fetch_cache(idx):
    with open(os.path.join(PLAN_CACHE_DIR, f'{idx}.dill'), 'rb') as f:
        return dill.load(f)
def _store_cache(idx, result):
    assert not os.path.exists(os.path.join(PLAN_CACHE_DIR, f'{idx}.dill'))
    with open(os.path.join(PLAN_CACHE_DIR, f'{idx}.dill'), 'wb') as f:
        dill.dump(result, f)

planner_with_success = PlannerWithSuccess()
