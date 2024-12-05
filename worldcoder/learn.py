import os, sys
import copy
import argparse
import datetime
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt

from .envs import get_env, get_env_args, add_env_args
from .agent import Agent, get_agent_args, add_agent_args
from .utils.logging import set_logger

def get_args():
    parser = argparse.ArgumentParser(description = "Example usage: python learn.py TODO")
    add_env_args(parser)
    add_agent_args(parser)

    parser.add_argument('--length', "-l", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=1000, help="number of episodes to run")
    parser.add_argument("--load", type=str, default=None, help="load a checkpoint (world model) from a file (.py)")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument('--max_llm_requests', type=int, default=200, help="maximum number of requests to the LLM")
    parser.add_argument('--prv_env_id', type=int, default=0, help="previous environment id")
    arguments = parser.parse_args()
    if arguments.curri:
        arguments.max_llm_requests = int(arguments.max_llm_requests * 1.5)
    print(arguments)
    return arguments

def main():
    arguments = get_args()
    default_env_args = get_env_args(arguments)
    env_name_list = default_env_args.pop("env_name_list")
    env_args_list = [
        {**default_env_args, "env_name": env_name}
        for env_name in env_name_list
    ]
    agent_args = get_agent_args(arguments)
    hyper_parameters = {
        "env_args_list": env_args_list,
        "agent_args": agent_args,
        "general_args": {
            'length': arguments.length,
            'episodes': arguments.episodes,
            'seed': arguments.seed,
            'load': arguments.load,
        }
    }
    if arguments.load is not None:
        hyper_parameters["load"] = arguments.load

    # create a temporary file for just this experiment, timestamped with the date and time
    # timestamp = datetime.datetime.now().isoformat()
    abbr = lambda env_args_list: "-".join([env_args["env_name"] for env_args in env_args_list]) if len(env_args_list) > 2 else env_args_list[0]["env_name"]
    sys_args = '_'.join([v.strip('-').strip().replace(' ', '_').replace(',', '').replace('=', '_').replace('/', '_') if len(v) < 50 else hashlib.md5(v.encode()).hexdigest() for v in sys.argv[1:]])
    output_directory = f"experiment_outputs/{abbr(env_args_list)}/{sys_args}"
    os.system(f"mkdir -p {output_directory}")
    print("output directory:", output_directory)

    env = get_env(env_args_list[0],)
    agent = Agent(
        world_model=None,
        **agent_args,
    )
    if arguments.load is not None:
        agent.load_full(arguments.load)

    log_filename = f"{output_directory}/log.txt"
    set_logger(log_filename)

    os.system(f"mkdir -p {output_directory}/world_models")
    next_world_model_id = 0
    os.system(f"mkdir -p {output_directory}/gameplay")

    # save the hyper parameters
    with open(f"{output_directory}/hyper_parameters.json", "w") as f:
        json.dump(hyper_parameters, f)

    total_reward = [] # how much reward we got each episode
    # best_total_reward = [] # how much reward we got each episode
    was_model_updated_in_episode = [] # whether the world model was updated in each episode
    was_episode_successful = [] # whether the episode was successful (all boxes on targets)

    prv_env_id = arguments.prv_env_id
    for ei, env_args in enumerate(env_args_list):
        if agent.total_costs and sum(agent.total_costs['requests']) >= arguments.max_llm_requests:
            print("Reached the maximum number of LLM requests, stopping.")
            break
        # synthesis_args["env_metadata"] = env.metadata
        env = get_env(env_args,)
        agent.synthesis_options['env_metadata'] = env.metadata
        agent.reset()
        print('!'*20, f'Switching to environment {env_args["env_name"]}', '!'*20)
        # for env_id in range(arguments.episodes*ei, arguments.episodes*(ei+1)):
        for env_id in range(prv_env_id, prv_env_id+arguments.episodes):
            if agent.total_costs: print(f"Total llm calls: {sum(agent.total_costs['requests'])}")
            if agent.total_costs and sum(agent.total_costs['requests']) >= arguments.max_llm_requests:
                print("Reached the maximum number of LLM requests, stopping.")
                break
            print(len(was_episode_successful), arguments.episodes*len(env_args_list), was_episode_successful[-10:])
            if len(was_episode_successful)-prv_env_id > 10 and all(was_episode_successful[-10:]):
                prv_env_id = env_id
                print("All episodes in the last 10 environments were successful, stopping.")
                break
            print(f"Running episode {env_id} of {arguments.episodes*len(env_args_list)}, env {env_args['env_name']}, reseting with seed {int(env_id-prv_env_id)}")
            state, mission, info = env.reset(seed=int(env_id-prv_env_id))
            os.system(f"mkdir -p {output_directory}/gameplay/{env_id}")
            with open(f"{output_directory}/gameplay/{env_id}/env_args.json", "w") as f:
                json.dump(env_args, f)

            agent.reset()
            done = False
            total_reward.append(0)
            was_model_updated_in_episode.append(0)
            was_episode_successful.append(0)

            was_world_model_updated = agent.learn_by_planning(state, mission, env_name=env_args["env_name"], mcts_budget=info['mcts_budget'] if 'mcts_budget' in info else None)
            if was_world_model_updated:
                was_model_updated_in_episode[-1] = 1
                # save the world model
                world_model_path = f"{output_directory}/world_models/{next_world_model_id}_env{env_id}_step-1"
                os.makedirs(world_model_path, exist_ok=True)
                agent.save(world_model_path)
                next_world_model_id += 1
                print("world model dumped to", world_model_path)
                if sum(agent.total_costs['requests']) >= arguments.max_llm_requests:
                    print("Reached the maximum number of LLM requests, stopping.")
                    break

            for t in range(arguments.length):

                env.screenshot(f"{output_directory}/gameplay/{env_id}/{t:03d}.png")

                if done: break

                action = agent.act(state, mission, mcts_budget=info['mcts_budget'] if 'mcts_budget' in info else None)

                old_state = state
                new_state, reward, done, new_info = env.step(action)
                total_reward[-1] += reward
                if done: was_episode_successful[-1] = 1

                was_world_model_updated = agent.learn(old_state, mission, action, new_state, reward, done)
                if was_world_model_updated:
                    was_model_updated_in_episode[-1] = 1
                    # save the world model
                    world_model_path = f"{output_directory}/world_models/{next_world_model_id}_env{env_id}_step{t}"
                    os.makedirs(world_model_path, exist_ok=True)
                    agent.save(world_model_path)
                    next_world_model_id += 1
                    print("world model dumped to", world_model_path)
                    if sum(agent.total_costs['requests']) >= arguments.max_llm_requests:
                        print("Reached the maximum number of LLM requests, stopping.")
                        break

                state = copy.deepcopy(new_state)

            if agent.world_model is not None:
                print("current world model after training on the first", env_id+1, "levels")
                print(agent.world_model.source_code())

            # now make a movie of the screenshots
            os.system(f"ffmpeg -framerate 5 -pattern_type glob -i '{output_directory}/gameplay/{env_id}/*.png' -c:v libx264 -r 30 -pix_fmt yuv420p -y {output_directory}/gameplay/{env_id}/animation.mp4")

            # save the learning curve
            with open(f"{output_directory}/learning_curve.json", "w") as f:
                json.dump({"episode_return": total_reward,
                        # 'best_episode_return': best_total_reward,
                        "model_update": was_model_updated_in_episode,
                        "success": was_episode_successful}, f)
            assert len(total_reward) == len(was_model_updated_in_episode) == len(was_episode_successful)
            # make a plot of the learning curve, save it to learning_curve.pdf
            plt.plot(total_reward,)
            # plt.plot(best_total_reward, label="optimal")
            plt.xlabel("episode")
            plt.ylabel("return (total reward)")
            # put black tick marks at each point where the world model was updated
            for x in range(len(total_reward)):
                if was_model_updated_in_episode[x]:
                    plt.axvline(x, color="black")
            plt.legend()
            plt.savefig(f"{output_directory}/learning_curve.pdf")
            # close the figure
            plt.close()

if __name__ == '__main__':
    main()

