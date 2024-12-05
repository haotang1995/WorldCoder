import os, os.path as osp
import copy
import numpy as np
import dill

from .synthesizer import add_synthesis_args, get_synthesis_args
from .synthesizer import refine_world_model, guess_reward
from .synthesizer.evaluator import PlanEvaluator, SepPlanEvaluator
from .world_model import WorldModel
from .planner import planner_with_success, add_planner_args, get_planner_args

def add_agent_args(parser):
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--agent_seed', type=int, default=None,)
    parser.add_argument('--minimum_buffer_size', type=int, default=10)
    parser.add_argument('--minimum_env_buffer_size', type=int, default=0)
    parser.add_argument('--single_env_plan_buffer_flag', action='store_true', default=False)
    parser.add_argument('--no_few_shot_new_reward_flag', dest='few_shot_new_reward_flag', action='store_false', default=True)
    # parser.add_argument('--few_shot_new_reward_flag', action='store_true', default=False)
    parser.add_argument('--no_reset_key_missions', dest='reset_key_missions', action='store_false', default=True)
    add_synthesis_args(parser)
    add_planner_args(parser)
def get_agent_args(args):
    return {
        'epsilon': args.epsilon,
        'seed': args.agent_seed if args.agent_seed is not None else args.seed,
        'minimum_buffer_size': args.minimum_buffer_size,
        'minimum_env_buffer_size': args.minimum_env_buffer_size,
        'single_env_plan_buffer_flag': args.single_env_plan_buffer_flag,
        'few_shot_new_reward_flag': args.few_shot_new_reward_flag,
        'reset_key_missions': args.reset_key_missions,
        'planning_options': get_planner_args(args),
        'synthesis_options': get_synthesis_args(args),
    }

class Agent():
    def __init__(
        self, world_model=None, epsilon=0.1,
        single_env_plan_buffer_flag=False,
        minimum_buffer_size=10, minimum_env_buffer_size=0,
        few_shot_new_reward_flag=False,
        reset_key_missions=True,
        seed=0,
        synthesis_options={}, planning_options={},
    ):
        self.world_model = world_model
        self.np_rng = np.random.default_rng(seed=seed,)
        assert isinstance(self.np_rng, np.random.Generator)
        self.total_costs = None if (
            world_model is None or
            'synthesizer_logs' not in world_model.__dict__ or
            not isinstance(world_model.synthesizer_logs[0], dict) or
            'total_costs' not in world_model.synthesizer_logs[0]
        ) else world_model.synthesizer_logs[0]['total_costs']
        self.total_new_costs = None if (
            world_model is None or
            'synthesizer_logs' not in world_model.__dict__ or
            not isinstance(world_model.synthesizer_logs[0], dict) or
            'total_new_costs' not in world_model.synthesizer_logs[0]
        ) else world_model.synthesizer_logs[0]['total_new_costs']

        self.experience_buffer = dict() # (s,m,a,s',r,d) -> {'state': s, 'mission': m, ..., 'info': info, 'info_next',...}
        self.new_data_non_evaluated = False
        self.env_buffer = dict() # (s, mission) -> {'state': s, 'mission': mission, 'info': info, }
        self.key_missions = set() # {mission}
        self.mission_accomplished = set() # {mission}

        self.tried_to_plan = False

        self.epsilon = epsilon
        self.single_env_plan_buffer_flag = single_env_plan_buffer_flag
        self.minimum_buffer_size = minimum_buffer_size
        self.minimum_env_buffer_size = minimum_env_buffer_size
        self.reset_key_missions = reset_key_missions
        self.synthesis_options = copy.deepcopy(synthesis_options)
        self.synthesis_options['np_rng'] = self.np_rng
        self.planning_options = copy.deepcopy(planning_options)

        self.few_shot_new_reward_flag = few_shot_new_reward_flag
        self.crt_reward_code = dict() # (missions, code) -> {'missions': missions, 'code': code, 'experience': one_crt_experience} # one_crt_experience is a dict of (s, m, a, s', r, d)
        self.cur_env_name = None
        self.cur_env_missions = set()

        self.plan_evaluator = PlanEvaluator()
    def reset(self):
        self.plan = list()
        self.failed_plan_for_current_world_model = False
        self.synthesis_options['np_rng'] = self.np_rng
        if self.reset_key_missions:
            self.key_missions = set()
        self.tried_to_plan = False
        self.cur_step = 0

    def act(self, state, mission, mcts_budget=None,):
        # construct a plan, and return the first action in that plan
        # if there is no world model, return a random action
        # returns a random action w.p. epsilon, also
        self.cur_step += 1
        if self.world_model is None or self.np_rng.random() < self.epsilon:
            print("random action")
            self.plan = list()
            return self.np_rng.choice(state.get_valid_actions())
        if self.plan is not None and len(self.plan) > 0:
            print('had a plan', self.plan[0])
            return self.plan.pop(0)

        # otherwise, use the world model to plan
        planning_options = copy.deepcopy(self.planning_options)
        if mcts_budget and self.planning_options['method'].lower() == 'mcts':
            planning_options['budget'] = mcts_budget
        planning_options['budget'] = 3 * planning_options['budget']
        if not self.failed_plan_for_current_world_model:
            self.plan, success = planner_with_success(
                initial_state=state, mission=mission,
                world_model=self.world_model,
                **planning_options
            )
        else:
            success = False
        if not success:
            print('planning failed')
            self.plan = list()
            self.failed_plan_for_current_world_model = True
            return self.np_rng.choice(state.get_valid_actions())
        else:
            print('successfully planned', self.plan)
            _state, _mission = state, mission
            for ai, _action in enumerate(self.plan):
                _new_state, _reward, _done = self.world_model.predict([(_state, _mission, _action,)])[0]
                print(ai, _action, _reward, _done, _new_state-_state)
                _state = _new_state
            # assert _done and _reward > 0
            self.failed_plan_for_current_world_model = False
            return self.plan.pop(0)

    def evaluate_and_update(self, care_about_plan=True,):
        self.new_data_non_evaluated = False
        inconsistent = self.world_model is None
        if not inconsistent:
            for state, mission, action, new_state, reward, done in self.experience_buffer:
                if str(mission) not in self.key_missions: continue
                prediction = self.world_model.predict([(state, mission, action)])[0]
                if isinstance(prediction, Exception):
                    print('prediction failed', prediction)
                    inconsistent = True
                    break
                pred_state, pred_reward, pred_done = prediction
                if pred_state != new_state or abs(pred_reward - reward) > 1e-6 or pred_done != done:
                    print('prediction inconsistent', pred_state != new_state, pred_reward, reward, pred_done, done)
                    inconsistent = True
                    break
        if self.synthesis_options['plan_obj_flag'] and care_about_plan and not inconsistent:
            plan_result = self.plan_evaluator(*(list(self.world_model.source_code().values())), self.env_buffer, self.key_missions,)
            inconsistent = not plan_result['success_flag']
            if inconsistent:
                print('plan failed')
        if inconsistent: actually_updated = self.update_world_model()
        else: actually_updated = False
        if actually_updated:
            self.plan = list()
            self.failed_plan_for_current_world_model = False
        return inconsistent, actually_updated

    def learn(self, state, mission, action, new_state, reward, done,):
        """returns True if the world model was updated, False otherwise"""
        # print("learn", state, mission, action, new_state, reward, done, new_state-state)
        print('~'*80)
        print(f"learn at step {self.cur_step}:", mission, action, reward, done, new_state-state)

        self.key_missions.add(str(mission))
        if reward > 0 and done:
            self.mission_accomplished.add(str(mission))
        # if not self.new_data_non_evaluated and (state, mission, action, new_state, reward, done) in self.experience_buffer:
            # print('already seen this', len(self.experience_buffer))
            # return False # already learned this
        self.experience_buffer[(state, mission, action, new_state, reward, done)] = {
            'state': state, 'mission': mission, 'action': action,
            'state_next': new_state, 'reward': reward, 'done': done,
            # 'info': info, 'info_next': info_next
        }
        if len(self.experience_buffer) < self.minimum_buffer_size and self.cur_step < 2 * self.minimum_buffer_size:
            print('not enough data to learn anything', len(self.experience_buffer))
            self.new_data_non_evaluated = True
            return False # not enough data to learn anything

        inconsistent, actually_updated = self.evaluate_and_update(care_about_plan=not self.tried_to_plan)
        if not inconsistent:
            print(f'consistent for {len(self.experience_buffer)} experiences')
        if actually_updated:
            print(f'updated world model after {len(self.experience_buffer)} experiences')
            self.plan = list()
            self.failed_plan_for_current_world_model = False
        return actually_updated

    def learn_by_planning(self, state, mission, env_name=None, mcts_budget=None,):
        """returns True if the world model was updated, False otherwise"""
        print('~'*80)
        print("learn_by_planning", state, mission,)
        old_reward_size = len(self.world_model.source_code()['reward']) if self.world_model is not None else 0
        self.key_missions.add(str(mission))
        if mcts_budget and self.planning_options['method'].lower() == 'mcts':
            planning_options = copy.deepcopy(self.planning_options)
            planning_options['budget'] = mcts_budget
        else:
            planning_options = self.planning_options
        new_env_info = {(state, mission): {
            'state': state,
            'mission': mission,
            'info': None,
            'planning_options': planning_options,
            'env_name': env_name,
        }}

        actually_updated = False
        if self.few_shot_new_reward_flag and self.cur_env_name != env_name:
            self.cur_env_name = env_name
            self.cur_env_missions = set([mission,])
            self.experience_buffer = dict()
            self.env_buffer = dict()
            self.new_data_non_evaluated = False
        elif self.few_shot_new_reward_flag and mission not in self.cur_env_missions:
            self.cur_env_missions.add(mission)
        if self.world_model is not None and str(mission) not in self.world_model.source_code()['reward']:
            actually_updated = self.guess_reward_func(new_env_info)
            assert str(mission) in self.world_model.source_code()['reward']

        if not self.single_env_plan_buffer_flag:
            if (state, mission) in self.env_buffer: return actually_updated # already learned this
            self.env_buffer.update(new_env_info)
            if len(self.env_buffer) < self.minimum_env_buffer_size: return actually_updated # not enough data to learn anything
        else:
            self.env_buffer = new_env_info
        if len(self.experience_buffer) < self.minimum_buffer_size:
            self.new_data_non_evaluated = True
            return actually_updated # not enough data to learn anything

        inconsistent, _actually_updated = self.evaluate_and_update(care_about_plan=True)
        actually_updated = actually_updated or _actually_updated
        actually_updated = actually_updated or len(self.world_model.source_code()['reward']) > old_reward_size
        if actually_updated:
            self.plan = list()
            self.failed_plan_for_current_world_model = False
        return actually_updated

    def update_world_model(self):
        if self.synthesis_options['plan_obj_flag']:
            if len(filter_w_missions(self.experience_buffer, self.key_missions)) < 1: return False
            # if len(filter_w_missions(self.env_buffer, self.key_missions)) < self.minimum_env_buffer_size: return False
        synthesis_options = {
            k: (
                len(self.env_buffer) < self.minimum_env_buffer_size or v
                if k.startswith('plan') and k.endswith('flag')
                else v
            )
            for k, v in self.synthesis_options.items()
        }

        if self.world_model is not None:
            _init_transit_code = copy.deepcopy(self.world_model.source_code()['transit'].strip())
            _init_reward_code = copy.deepcopy(self.world_model.source_code()['reward'])
        else:
            _init_transit_code = None
            _init_reward_code = None
        fitness, (_transit_code, _reward_code, _output, path) = refine_world_model(
            _init_transit_code, _init_reward_code,
            self.experience_buffer,
            self.env_buffer,
            self.key_missions,
            **(synthesis_options),
        )
        self.world_model = WorldModel(_transit_code, _reward_code, (_output, path))
        self.plan = list()
        self.failed_plan_for_current_world_model = False
        if self.total_costs is None:
            self.total_costs = self.world_model.synthesizer_logs[0]['total_costs']
        else:
            for k, v in self.world_model.synthesizer_logs[0]['total_costs'].items():
                self.total_costs[k] += v
        if self.total_new_costs is None:
            self.total_new_costs = self.world_model.synthesizer_logs[0]['total_new_costs']
        else:
            for k, v in self.world_model.synthesizer_logs[0]['total_new_costs'].items():
                self.total_new_costs[k] += v
        # print('mean costs for agent2', {k: np.mean(v) for k, v in self.total_costs.items()})
        print('sum costs for agent2', {k: np.sum(v) for k, v in self.total_costs.items()})
        print('sum new costs for agent2', {k: np.sum(v) for k, v in self.total_new_costs.items()})
        self.tried_to_plan = True
        return True

    def guess_reward_func(self, new_env_info):
        if self.world_model is None:
            return False
        # if len(self.mission_accomplished) < 1:
            # return False
        self.mission_accomplished = set(self.world_model.source_code()['reward'].keys())
        reward_code = copy.deepcopy(self.world_model.source_code()['reward'])
        str_missions = {str(m) for m in self.mission_accomplished}
        reward_code = {k: v for k, v in reward_code.items() if str(k) in str_missions}
        mission = str(list(new_env_info.values())[0]['mission'])
        if mission in reward_code:
            return False
        _code, _output = guess_reward(
            reward_code,
            mission,
            **self.synthesis_options,
        )
        reward_code[mission] = _code
        self.world_model = WorldModel(
            self.world_model.source_code()['transit'],
            reward_code,
            (_output, None),
        )
        if self.total_costs is None:
            self.total_costs = self.world_model.synthesizer_logs[0]['total_costs']
            self.total_new_costs = self.world_model.synthesizer_logs[0]['total_new_costs']
        else:
            for k, v in self.world_model.synthesizer_logs[0]['total_costs'].items():
                self.total_costs[k] += v
            for k, v in self.world_model.synthesizer_logs[0]['total_new_costs'].items():
                self.total_new_costs[k] += v
        # print('mean costs for agent2', {k: np.mean(v) for k, v in self.total_costs.items()})
        print('sum costs for agent2', {k: np.sum(v) for k, v in self.total_costs.items()})
        print('sum new costs for agent2', {k: np.sum(v) for k, v in self.total_new_costs.items()})
        return True

    def save(self, path):
        if self.world_model is None:
            return
        self.world_model.save(path)
        with open(osp.join(path, 'total_costs.pkl'), 'wb') as f:
            dill.dump(self.total_costs, f)
        with open(osp.join(path, 'total_new_costs.pkl'), 'wb') as f:
            dill.dump(self.total_new_costs, f)
        with open(osp.join(path, 'experience_buffer.pkl'), 'wb') as f:
            dill.dump(self.experience_buffer, f)
        with open(osp.join(path, 'env_buffer.pkl'), 'wb') as f:
            dill.dump(self.env_buffer, f)
    def load_full(self, path):
        self.world_model = WorldModel.load(path)
        if 'total_costs.pkl' in os.listdir(path) and 'total_new_costs.pkl' in os.listdir(path) and 'experience_buffer.pkl' in os.listdir(path) and 'env_buffer.pkl' in os.listdir(path):
            with open(osp.join(path, 'total_costs.pkl'), 'rb') as f:
                self.total_costs = dill.load(f)
            with open(osp.join(path, 'total_new_costs.pkl'), 'rb') as f:
                self.total_new_costs = dill.load(f)
            with open(osp.join(path, 'experience_buffer.pkl'), 'rb') as f:
                self.experience_buffer = dill.load(f)
            with open(osp.join(path, 'env_buffer.pkl'), 'rb') as f:
                self.env_buffer = dill.load(f)
        else:
            assert 'synthesizer_logs' in self.world_model.__dict__
            self.experience_buffer = self.world_model.synthesizer_logs[0]['exp_result']['experiences']
            self.env_buffer = self.world_model.synthesizer_logs[0]['plan_result']['envs']
def filter_w_missions(experiences, key_missions):
    return SepPlanEvaluator._filter_w_missions(experiences, key_missions)
