import gym
import logging
import sys

class RandNonLearner:
    

LEARNER = RandNonLearner

def run_one_episode(env, model):
    agent = model.new_agent()
    last_observation = env.reset()
    env.render()
    done = False
    total_reward = 0.0
    while not done:
        last_observation, reward, done, unused = env.step(env.action_space.sample())
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
  model = LEARNER.new_model()
  while True:
      model = run_one_episode(env, model)
