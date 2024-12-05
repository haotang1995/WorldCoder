#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp
import time

from .main import PlanCache, Session, ENGINE, select
from sqlalchemy import delete

def main():
    cache_schema = PlanCache
    cache_schema.metadata.create_all(ENGINE)

    session = Session(ENGINE)
    session.begin()

    stmt = select(cache_schema.idx).order_by(cache_schema.idx)
    generations = session.execute(stmt).fetchall()
    assert generations, f"generations: {generations}"
    max_idx = max([row[0] for row in generations]) if generations else 0
    print(f"max_idx: {max_idx}")

    stmt = select(cache_schema.idx).where(cache_schema.idx == max_idx)
    generations = session.execute(stmt).fetchall()
    assert generations, f"generations: {generations}"
    assert len(generations) > 1, f"generations: {generations}"

    stmt = delete(cache_schema).where(cache_schema.idx == max_idx)
    session.execute(stmt)
    session.commit()
    session.close()

    os.remove(osp.join(osp.dirname(osp.abspath(__file__)), 'plan_cache', f'{max_idx}.dill'))

if __name__ == '__main__':
    while True:
        try:
            main()
        except Exception as e:
            print(e)
        time.sleep(60)
