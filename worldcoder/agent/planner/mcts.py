#!/usr/bin/env python
# coding=utf-8

import time
import numpy as np
import nltk
from nltk.corpus import stopwords

class MCTSNode:
    def __init__(
        self,
        # Global variables
        mission, world_model, fake_reward_func, visited_states, rng, max_depth, ucb_c,
        # Node variables
        depth, state, parent, action, true_reward, fake_reward, done, visited_flag,
    ):
        self.mission = str(mission)
        self.world_model = world_model
        self.fake_reward_func = fake_reward_func
        self.visited_states = visited_states
        self.rng = rng
        self.max_depth = max_depth
        self.ucb_c = ucb_c

        self.depth = depth
        self.state = state
        self.parent = parent
        self.action = action
        self.reward = true_reward + fake_reward
        self.true_reward = true_reward
        self.fake_reward = fake_reward
        self.done = done
        self.visited_flag = visited_flag

        # # Just for 'put a hot tomato in fridge'
        # if self.parent:
            # entities = self.parent.state.to_pyrunnable()
            # agent = [obj for obj in entities if obj.is_entity_type('agent')][0]
            # if agent.holding and 'tomato' == agent.holding.strip('0123456789'):
                # holded_obj = [obj for obj in entities if obj.name == agent.holding][0]
                # if not holded_obj.ishot:
                    # for recp in [obj for obj in entities if obj.is_entity_type('microwave')]:
                        #, reverse=True if agent.loc == recp.loc:
                            # assert [act for act in self.parent.valid_actions if 'heat' in str(act).lower()], self.path()
                            # assert [act for act in self.parent.valid_actions if 'goto(dest=fridge' in str(act).lower()], self.path()
                            # break
                # else:
                    # for recp in [obj for obj in entities if obj.is_entity_type('fridge')]:
                        # if agent.loc == recp.loc:
                            # assert [act for act in self.parent.valid_actions if 'put' in str(act).lower()], self.path()
                            # print(self.parent.visits, self.path(), self.true_reward, self.done, self.parent.valid_actions, recp.openable, recp.isopen)
                            # if 'fridge' in str(self.action) and 'put' in str(self.action):
                                # assert False, (self.path(), self.reward, self.done)
                            # break

        # self.valid_actions = list(sorted(list(set(state.get_valid_actions())), key=str))
        # self.rng.shuffle(self.valid_actions)
        mission_set = set(self.fake_reward_func.remove_stopwords(self.mission))
        if state is not None:
            self.valid_actions = list(sorted(list(set(state.get_valid_actions())), key=lambda act: len(mission_set & set(self.fake_reward_func.remove_stopwords(str(act)))), reverse=True))
        self.children = dict()
        self.visits = 1
        self.qvalue = self.reward
        self._all_terminated = self.is_terminal() or self.done
    def is_fully_expanded(self):
        return len(self.children) == len(self.valid_actions)
    def is_terminal(self):
        return self.state is None or self.done is None or self.visited_flag or self.depth >= self.max_depth or not len(self.valid_actions)
    def success(self):
        return self.done and self.reward > 1e-6
    def expand(self):
        assert not self.is_fully_expanded()
        action = self.valid_actions[len(self.children)]
        pred = self.world_model.predict([(self.state, self.mission, action)])[0]
        if isinstance(pred, (list, tuple)) and len(pred) == 3:
            new_state, reward, done = pred
            # Only to test for `put a pen in shelf`
            # true_reward = 0
            # for obj in new_state.to_pyrunnable():
                # if obj.is_entity_type('pen') and obj.in_on and 'shelf' in obj.in_on:
                    # true_reward = 1
                    # break
            # assert true_reward == reward, (new_state, true_reward, reward)
            visited_flag = new_state in self.visited_states
            self.visited_states.add(new_state)
            fake_reward = self.fake_reward_func(self.state, action, new_state, self,)
        else:
            new_state, reward, done = None, 0, None
            fake_reward = 0
            visited_flag = True
        self.children[action] = MCTSNode(
            self.mission, self.world_model, self.fake_reward_func, self.visited_states, self.rng, self.max_depth, self.ucb_c,
            self.depth + 1, new_state, self, action, reward, fake_reward, done, visited_flag,
        )
        return self.children[action]
    def best_child(self,):
        return max(
            self.children.values(),
            key=lambda child: -1e10*child.all_terminated() + child.qvalue / child.visits + self.ucb_c * (2 * np.log(self.visits) / child.visits) ** 0.5
        )
    def backpropagate(self):
        node = self
        # value = -0.1 * node.is_terminal() + node.reward
        value = node.reward
        while node.parent is not None:
            node.parent.qvalue += value
            node.parent.visits += 1
            node = node.parent
            # value += node.reward

    def all_terminated(self):
        if self._all_terminated:
            return True
        if self.is_fully_expanded():
            self._all_terminated = all(child.all_terminated() for child in self.children.values())
        return self._all_terminated
    def path(self):
        if self.parent is None:
            return []
        return self.parent.path() + [self.action]

class BM25RewardFunc:
    nltk.download('stopwords')
    nltk.download('punkt')
    def __init__(self, mission):
        self.mission = str(mission)
        self.mission = set(self.remove_stopwords(self.mission))

        self.doc_cnt = dict() # For idf
        self.total_doc = 0
        self.avgdl = 0

        self.k1, self.b = 1.5, 0.75
    def __call__(self, state, action, new_state, node,):
        diff = str(action) + '\n' + (new_state - state)
        while node.parent:
            diff = str(node.action) + '\n' + (node.state - node.parent.state) + '\n' + diff
            node = node.parent
        diff = self.remove_stopwords(diff)

        for w in set(diff):
            if w not in self.doc_cnt:
                self.doc_cnt[w] = 0
            self.doc_cnt[w] += 1
        self.total_doc += 1
        self.avgdl = self.avgdl * ((self.total_doc - 1) / self.total_doc) + len(diff) / self.total_doc

        fake_reward = 0
        for mw in self.mission:
            if mw in diff:
                count = diff.count(mw)
                idf = np.log((self.total_doc - self.doc_cnt[mw] + 0.5) / (self.doc_cnt[mw] + 0.5) + 1)
                # assert idf > 0, (self.total_doc, self.doc_cnt[mw])
                fake_reward += idf * count * (self.k1 + 1) / (count + self.k1 * (1 - self.b + self.b * len(diff) / self.avgdl))
                # assert fake_reward >= 0, (fake_reward, mw, diff)
        fake_reward /= len(self.mission)
        # fake_reward = len(diff & self.mission) / len(self.mission)
        # diff = ' '.join(self.remove_stopwords(diff))
        # fake_reward2 = sum([mw in ' '.join(diff) for mw in self.mission])/len(self.mission) # To compensate for the difference between pencil & pencil1
        # assert fake_reward == fake_reward2, (fake_reward, fake_reward2, diff, self.mission)
        return fake_reward
    @classmethod
    def remove_stopwords(cls, text):
        text = text.lower()
        for d in '0123456789' + '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~':
            text = text.replace(d, ' ')
        words = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        return [w for w in words if w not in stop_words]

def mcts(
    state, mission, world_model, fake_reward_func=None,
    budget=30000, max_depth=30, ucb_c=1, seed=0,
):
    if fake_reward_func is None:
        fake_reward_func = BM25RewardFunc(mission)
    visited_states = {state,}
    rng = np.random.default_rng(seed=seed,)
    root = MCTSNode(
        mission, world_model, fake_reward_func, visited_states, rng, max_depth, ucb_c,
        0, state, None, None, 0, 0, False, False,
    )

    _visited_states_just_for_debug = [state,]
    new_node_num = 0
    start_time = time.time()
    while new_node_num < budget:
        node = root
        _expanded=False
        while not node.is_terminal():
            if not node.is_fully_expanded():
                node = node.expand()
                if node.success():
                    print(f'Found success! in {time.time()-start_time} seconds with {new_node_num} new nodes, path length: {len(node.path())}, path: {node.path()}, node.reward, node.done,')
                    return node.path(), True
                new_node_num += 1
                _expanded=True
                if node.state is not None:
                    if not node.visited_flag:
                        _visited_states_just_for_debug.append(node.state)
                    old_node_index = _visited_states_just_for_debug.index(node.parent.state)
                    new_node_index = _visited_states_just_for_debug.index(node.state)
                    if new_node_num % 3000 == 0:
                        print(f'new_node_num: {new_node_num}, from node: {old_node_index} at depth {node.parent.depth}, to node: {new_node_index} at depth {node.depth}, used time: {time.time()-start_time}')
                        print(f'visit new state: {node.state-node.parent.state}; action: {node.action}; reward: {node.reward}; done: {node.done}; fake_reward: {node.fake_reward}; visited_flag: {node.visited_flag}')
                break;
            else:
                node = node.best_child()
        if not _expanded:
            index = _visited_states_just_for_debug.index(node.state)
            print(f'Terminal node: {index}, all_terminated: {node.all_terminated()}, is_terminal: {node.is_terminal()}, visited_flag: {node.visited_flag}, done: {node.done}, reward: {node.reward}')
        node.backpropagate()
        if root.all_terminated():
            print('All terminated')
            break
    if not root.all_terminated():
        print('Budget exhausted after', new_node_num, 'new nodes')

    return node.path(), False

