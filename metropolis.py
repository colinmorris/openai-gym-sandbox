# Sampling better models using the Metropolis algorithms.

import gym
import logging
import math
import random
import sys

class OneActionAgent(object):
    def __init__(self, action):
        self.action = action

    def act(self):
        return self.action

class OneActionModel(object):
    def __init__(self, env):
        self.env = env
        self.action = env.action_space.sample()

    def symmetric_mutate(self):
        return OneActionModel(env)

    def new_agent(self):
        return OneActionAgent(self.action)

LEARNER = OneActionModel

def reward_to_value(reward):
    return math.exp(reward)

def run_one_episode(env, model):
    agent = model.new_agent()
    last_observation = env.reset()
    env.render()
    done = False
    total_reward = 0.0
    while not done:
        last_observation, reward, done, unused = \
            env.step(agent.act(last_observation))
        total_reward += reward
        env.render()
    print total_reward
    return total_reward

if __name__ == "__main__":
  try:
    env_name = sys.argv[1]
  except IndexError:
    env_name = "Copy-v0"
    logging.warning("No environment name provided. Defaulting to {}".format(env_name))
  env = gym.make(env_name)
  model = LEARNER(env)
  previous_value = reward_to_value(run_one_episode(env, model))
  while True:
      candidate_model = model.symmetric_mutate()
      reward = run_one_episode(env, candidate_model)
      value = reward_to_value(reward)
      if value / previous_value > random.rand():
          model = candidate_model
          previous_value = value
