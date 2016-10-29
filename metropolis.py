"""Sampling better models using the Metropolis algorithms.

Tested on...
  Cartpole (success)
  MountainCar (partial success)

TODO:
  - pull out magic numbers into named constants
  - Some kind of annealing. Probably want to vary number of runs we average
    over.
  - generalize so this works with environments with different obs/action
    spaces (e.g. the algorithmic envs)
"""

import gym
import numpy as np
import logging
import math
import random
import sys
from matplotlib import pyplot as plt
import itertools as it
import argparse

class LinearThreshModel(object):
    """A network with linear threshold units (up to the final softmax layer)
    """
    @classmethod
    def new_model(kls, env):
        if isinstance(env.observation_space, gym.spaces.box.Box):
            boxshape = env.observation_space.shape
            assert len(boxshape) == 1
            input_size = boxshape[0]
        elif isinstance(env.observation_space, gym.spaces.discrete.Discrete):
            input_size = env.observation_space.n
        else:
            assert False
        layer_sizes = [input_size, env.action_space.n]
        weights = []
        biases = []
        for i in range(1, len(layer_sizes)):
            weights.append( np.zeros( (layer_sizes[i-1], layer_sizes[i]) ) )
            biases.append( np.zeros( (layer_sizes[i],) ) )
        return kls(env, weights, biases)

    def __init__(self, env, weights, biases):
        self.env = env
        self.input_size = weights[0].shape[0]
        nactions = env.action_space.n 
        self.weights = weights
        self.biases = biases

    def symmetric_mutate(self):
        # Sometimes mutations are very small in magnitude, sometimes they're very large
        m = math.exp(np.random.normal() * 4 - 4)
        new_weights = [w + m * np.random.normal(size = w.shape) for w in self.weights]
        new_biases = [b + m * np.random.normal(size = b.shape) for b in self.biases]
        new_model = LinearThreshModel(self.env, new_weights, new_biases)
        return new_model

    def new_agent(self):
        return LinearThreshAgent(self.env, self)

    def debug_string(self):
        return `self.weights` + " :: " + `self.biases`

    def regularization_cost(self):
        return sum([np.sum(w ** 2) for w in (self.weights + self.biases)]) * 0.01

class LinearThreshAgent(object):
    def __init__(self, env, model):
        self.env = env
        self.model = model
    
    def act(self, obs):
        if isinstance(obs, int):
            # one-hot encoding
            input_layer = np.zeros(model.input_size)
            input_layer[obs] = 1
        elif isinstance(obs, np.ndarray):
            input_layer = obs
        else:
            assert False
        prev = input_layer
        for i, (w, b) in enumerate(zip(self.model.weights, self.model.biases)):
            if i == len(self.model.weights)-1:
                # final softmax layer
                prev = np.dot(prev, w) + b
            else:
                prev = np.dot(prev, w) > b
        return sample_softmax(prev)

def sample_softmax(sm):
    sm = sm - np.max(sm)
    probs = np.exp(sm)
    probs = probs/np.sum(probs)
    try:
        return np.random.choice(a = np.arange(len(sm)), p = probs)
    except ValueError, e:
        logging.warning("softmax failure")
        return 0
      
LEARNER = LinearThreshModel

def compute_value(reward, regularization_cost):
    return math.exp(reward / 10.0 - regularization_cost)

def run_one_episode(env, model, render):
    agent = model.new_agent()
    last_observation = env.reset()
    if render:
        env.render()
    done = False
    total_reward = 0.0
    i = 0
    # We need some time limit for environments like MountainCar that can continue indefinitely
    while not done and i < 2000:
        i += 1
        last_observation, reward, done, unused = \
            env.step(agent.act(last_observation))
        total_reward += reward
        if render:
            env.render()
    return total_reward

def plot_rewards(rewards, switch_indices):
    plt.clf()
    last = 0
    switch_pts = []
    switch_indices, si2 = it.tee(switch_indices)
    si2.next()
    for (swindex, next_swindex) in it.izip_longest(switch_indices,si2, fillvalue=len(rewards)):
        for i in range(swindex, next_swindex):
            switch_pts.append(rewards[swindex])
        last = swindex
    plt.plot(rewards, color='blue', marker='.', linestyle='None', alpha=.5)
    #plt.plot(switch_indices, switch_pts, color='red', marker='.', linestyle='None') 
    plt.plot(switch_pts, color='red', linestyle=':')
    plt.savefig('rewards.png')
    print "wrote rewards"

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('env', nargs='?', default='CartPole-v0')
  parser.add_argument('--monitor')
  args = parser.parse_args()
  env = gym.make(args.env)
  if args.monitor:
    env.monitor.start(args.monitor, force=True)
  model = LEARNER.new_model(env)
  previous_value = None
  episode = 0
  mean_rewards = []
  switch_indices = []
  while True:
      candidate_model = model.symmetric_mutate()
      rewards = [run_one_episode(env, candidate_model, False) for _ in range(5)]
      reward = np.mean(rewards)
      if reward >= env.spec.reward_threshold:
        print "Reached reward threshold!"
        rewards2 = [run_one_episode(env, candidate_model, False) for _ in range(env.spec.trials)]
        if np.mean(rewards2) >= env.spec.reward_threshold:
          break
        else:
          print "Oops, guess it was a fluke"
      mean_rewards.append(reward)
      value = compute_value(reward, candidate_model.regularization_cost())
      print "{} reward: {} (std={:.2f}; value: {}".format(episode, reward, np.std(rewards), value)
      if previous_value is None or value / previous_value > random.random():
          model = candidate_model
          previous_value = value
          print "switched; new model: {}".format(model.debug_string())
          switch_indices.append(episode)
      episode += 1
      if (episode % 80) == 0:
        run_one_episode(env, model, True)
        plot_rewards(mean_rewards, switch_indices)
