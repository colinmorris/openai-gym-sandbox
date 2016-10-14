import gym
import time
import logging
import itertools
import sys

NO_OUTPUT = -1

class PolicyEnumerator(object):

  def __init__(self, env):
    assert isinstance(env, gym.envs.algorithmic.algorithmic_env.AlgorithmicEnv)
    self.env = env
    # directions
    dirs, write_mode, write_chars = self.env.action_space.spaces
    self.dirs = range(dirs.n)
    self.chars = range(write_chars.n)
    self.n_chars = write_chars.n
    self.n_inputs = self.n_chars+1 # plus blank space
    self.n_outputs = self.n_chars+1 # plus "don't write"
    # All characters plus a null/blank character
    self.chars_plus = [-1] + self.chars

  def enum(self, max_states):
    for n in range(1, max_states+1):
      for policy in self.enumerate_policies_with_nstates(n):
        yield policy

  def enumerate_policies_with_nstates(self, n):
    direction_pols = self.enumerate_direction_policies(n)
    output_pols = self.enumerate_output_policies(n)
    state_pols = self.enumerate_state_policies(n)
    expo = n * self.n_inputs
    ndpols = len(self.dirs)**expo
    nopols = self.n_outputs**expo
    nspols = n**expo
    logging.info("N policies for {} states:\ndirection:{:,} output:{:,} state:{:,}\t\
      total:{:,}".format(n, ndpols, nopols, nspols,
      n**expo, ndpols*nopols*nspols)
    )
    # TODO: should do some kind of diagonal enumeration? Or maybe it doesn't 
    # matter that much.

    # Originally used itertools.product, but surprisingly it was *really* slow
    # and memory-intensive compared to nested loops
    for dp in direction_pols:
      for op in self.enumerate_output_policies(n):
      #for op in output_pols:
        for sp in self.enumerate_state_policies(n):
        #for sp in state_pols:
          yield AlgorithmicPolicy(dp, op, sp)

  def enumerate_direction_policies(self, n_states):
    return self._enumerate_subpolicies(self.dirs, n_states)

  def enumerate_output_policies(self, n_states):
    return self._enumerate_subpolicies(self.chars_plus, n_states)
  
  def enumerate_state_policies(self, n_states):
    # minor optimization
    if n_states == 1:
      yield lambda *args : 0
    else:
      for pol in self._enumerate_subpolicies(range(n_states), n_states):
        yield pol

  def _enumerate_subpolicies(self, codomain, n_states):
    states = range(n_states)
    # Start with policies just based on the current state, ignoring input char
    images = itertools.product(codomain, repeat=n_states)
    for image in images:
      #yield lambda obs, state: image[state]
      yield simple_policy_factory(image)
    # Now policies based on current state and input char. Generate all
    # n-ary strings of length (n_states*n_inputs)
    images = itertools.product(codomain, repeat=(n_states*self.n_inputs))
    for image in images:
      yield policy_factory(image, self.n_inputs)
      # Can't do this because blah blah variable binding 
      #yield lambda obs, state: image[(state*self.n_inputs)+obs]

def simple_policy_factory(image):
  return lambda obs, state: image[state]
def policy_factory(image, n):
  return lambda obs, state: image[(state*n)+obs]
      
class AlgorithmicPolicy(object):

  def __init__(self, dp, op, sp):
    self.direction_policy, self.output_policy, self.state_policy = dp, op, sp

  def get_action(self, obs, state):
    direction = self.direction_policy(obs, state)
    output = self.output_policy(obs, state)
    output_tuple = (0, 0) if output == NO_OUTPUT else (1, output)
    return (direction,) + output_tuple
  
  def run(self, env):
    seen_reward_thresh = False
    reward_countdown = env.spec.trials
    while reward_countdown:
      success, reward = self.run_episode(env)
      if not success:
        return False
      seen_reward_thresh = reward >= env.spec.reward_threshold
      if (seen_reward_thresh):
        reward_countdown -= 1
    return reward_countdown == 0

  def run_episode(self, env):
    obs = env.reset()
    done = False
    total_reward = 0
    state = 0
    while not done:
      action = self.get_action(obs, state)
      state = self.state_policy(obs, state)
      obs, reward, done, _ = env.step(action)
      total_reward += reward
    # Assumption (which should hold for all algorithmic envs): an episode is 
    # overall successful iff the last step has positive reward
    return reward > 0, total_reward

def solve_env(env, max_states):
  pols = PolicyEnumerator(env).enum(max_states)
  for i, pol in enumerate(pols):
    success = pol.run(env)
    if success:
      return True
    if (i % 100000) == 0:
      logging.debug('i={:,}'.format(i))
  return False

if __name__ == '__main__':
  try:
    env_name = sys.argv[1]
  except IndexError:
    env_name = 'Copy-v0'
    logging.warning("No environment name provided. Defaulting to {}".format(env_name))
  env = gym.make(env_name)
  t0 = time.time()
  max_states = 1
  succ = solve_env(env, max_states)
  elapsed = time.time() - t0
  print "{} after {:.1f}s".format(
    "Solved" if succ else "Exhausted policies", elapsed
  )
