import multiprocessing as mp

import random
import copy
import argparse
import os, os.path as osp
import json
import numpy as np
import matplotlib.pyplot as plt

from .envs import get_env, get_env_args, add_env_args
from .agent import Agent, get_agent_args, add_agent_args
from .utils.logging import set_logger

def get_args():
    parser = argparse.ArgumentParser()
    add_env_args(parser)
    add_agent_args(parser)
    parser.add_argument("--key_words", type=str, nargs="+", default=None, help="key words to filter the environments")
    parser.add_argument('--length', "-l", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=10, help="number of episodes to run")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    arguments = parser.parse_args()
    return arguments

def evaluate(world_model_path, args):
    print("!"*20, "Evaluating", world_model_path, "!"*20)
    assert isinstance(args, argparse.Namespace)
    assert isinstance(world_model_path, str)
    assert osp.exists(world_model_path)
    world_model_path = osp.abspath(world_model_path)
    env_np_rng = np.random.default_rng(args.seed)
    env_seeds = env_np_rng.integers(0, 2**32, args.episodes)
    env_id = [n for n in osp.basename(world_model_path).split("_") if n.startswith("env")]
    assert len(env_id) == 1
    env_id = int(env_id[0][3:])
    gameplay_path = osp.join(osp.dirname(osp.dirname(world_model_path)), "gameplay", str(env_id))
    assert osp.exists(gameplay_path), gameplay_path
    with open(osp.join(gameplay_path, "env_args.json"), "r") as f:
        env_args = json.load(f)
    env = get_env(env_args)
    output_directory = osp.join(world_model_path, "eval_game_play")
    os.makedirs(output_directory, exist_ok=True)
    eps_returns_path = osp.join(output_directory, "eps_returns.json")
    if osp.exists(eps_returns_path):
        with open(eps_returns_path, "r") as f:
            data = json.load(f)
            eps_returns = data["episode_return"]
            success = data["success"]
        if len(eps_returns) >= args.episodes:
            print("Already evaluated, skipping")
            return

    agent_args = get_agent_args(args)
    agent_args['epsilon'] = 0.
    agent = Agent(
        world_model=None,
        **get_agent_args(args),
    )
    agent.load_full(world_model_path)
    total_reward = eps_returns if osp.exists(eps_returns_path) else [] # total reward for each episode
    was_episode_successful = success if osp.exists(eps_returns_path) else [] # 1 if the episode was successful, 0 otherwise
    init_env_id = len(total_reward)
    env = get_env(env_args,)
    print('~'*20, f'Evaluating in environment {env_args["env_name"]}', '~'*20)
    for env_id in range(init_env_id, args.episodes):
        print(f"Running episode {env_id} of {args.episodes}, reseting with seed {env_seeds[env_id]}")
        state, mission, info = env.reset(seed=int(env_seeds[env_id]))
        os.makedirs(os.path.join(output_directory, f"{env_id}"), exist_ok=True)

        agent.reset()
        done = False
        total_reward.append(0)
        was_episode_successful.append(0)
        for t in range(args.length):
            env.screenshot(f"{output_directory}/{env_id}/{t:03d}.png")

            if done: break

            action = agent.act(state, mission,)

            old_state = state
            new_state, reward, done, new_info = env.step(action)
            total_reward[-1] += reward
            if done: was_episode_successful[-1] = 1

            state = copy.deepcopy(new_state)

        # now make a movie of the screenshots
        os.system(f"ffmpeg -framerate 5 -pattern_type glob -i '{output_directory}/{env_id}/*.png' -c:v libx264 -r 30 -pix_fmt yuv420p -y {output_directory}/{env_id}/animation.mp4")

        # save the learning curve
        with open(eps_returns_path, "w") as f:
            json.dump({"episode_return": total_reward,
                       "success": was_episode_successful}, f)
        assert len(total_reward) == len(was_episode_successful)
        # make a plot of the learning curve, save it to learning_curve.pdf
        plt.plot(total_reward,)
        # plt.plot(best_total_reward, label="optimal")
        plt.xlabel("episode")
        plt.ylabel("return (total reward)")
        plt.legend()
        plt.savefig(f"{output_directory}/eps_returns.pdf")
        # close the figure
        plt.close()

def main():
    args = get_args()
    curdir = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'experiment_outputs')
    paths = [
        osp.join(curdir, env_name, time_id, 'world_models', wm_id)
        for env_name in os.listdir(curdir) if osp.isdir(osp.join(curdir, env_name)) and env_name not in ['history'] and (not args.key_words or any(kw.lower() in env_name.lower() for kw in args.key_words))
        for time_id in os.listdir(osp.join(curdir, env_name)) if osp.isdir(osp.join(curdir, env_name, time_id)) and osp.exists(osp.join(curdir, env_name, time_id, 'world_models'))
        for wm_id in os.listdir(osp.join(curdir, env_name, time_id, 'world_models')) if osp.isdir(osp.join(curdir, env_name, time_id, 'world_models', wm_id))
    ]
    random.shuffle(paths)
    paths = [
        p for p in paths if any([kw in p for kw in [
            '/home/ht383/WorldCoder/experiment_outputs/pick_place-heat_place-cool_place-clean_place-examine/env_alfworld_seed_0_no_verbose_max_llm_requests_500_planner-budget_6000/',
        ]])
    ]
    print(f'Found {len(paths)} paths')
    print(paths)
    for path in paths:
        evaluate(path, args)
    # with mp.Pool(mp.cpu_count()//10) as p:
        # p.starmap(evaluate, [(path, args) for path in paths])

if __name__ == '__main__':
    main()

