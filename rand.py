# Take random actions always.

import gym
import logging
import sys

def run_one_episode(env):
    unused_observation = env.reset()
    env.render()
    done = False
    total_reward = 0.0
    while not done:
        unused_observation, reward, done, unused = env.step(env.action_space.sample())
        total_reward += reward
        env.render()
    print total_reward

if __name__ == '__main__':
  try:
    env_name = sys.argv[1]
  except IndexError:
    env_name = 'Copy-v0'
    logging.warning("No environment name provided. Defaulting to {}".format(env_name))
  env = gym.make(env_name)
  while True:
      run_one_episode(env)
