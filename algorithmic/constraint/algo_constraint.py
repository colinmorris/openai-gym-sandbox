import gym
import time
import logging
import itertools
import sys
from pysmt.shortcuts import Symbol, Or, And
import pysmt.shortcuts as sc
import pysmt.typing as tp
from pysmt.solvers.solver import Solver

NO_OUTPUT = -1

class AlgorithmicSolver(object):

  def __init__(self, env, states):
    assert isinstance(env, gym.envs.algorithmic.algorithmic_env.AlgorithmicEnv)
    self.env = env
    # directions
    dirs, write_mode, write_chars = self.env.action_space.spaces
    assert dirs.n == 2
    self.dirs = "left", "right"
    self.n_chars = write_chars.n
    self.chars = [ chr(ord('A')+i) for i in range(write_chars.n) ]
    self.n_inputs = self.n_chars+1 # plus blank space
    self.n_outputs = self.n_chars+1 # plus "don't write"
    # All characters plus a null/blank character
    self.chars_plus = self.chars + ['_']
    self.n_states = states
    self.states = range(self.n_states)
    self.solver = Solver()

  def solve(self):
    while 1:
      self.helper.reset_dirty_rules()
      done, success = self.try_model()
      if done:
        return success

  def try_model(self):
    max_eps = 1000 # failsafe
    i = 0
    while reward_countdown and i < max_eps:
      i+= 1
      success, reward = self.run_episode()
      if not success:
        return False
      if reward >= env.spec.reward_threshold:
        return True
    logging.warning("Performed {} iters without failure or reaching reward\
      threshold. Sus.".format(max_eps))
    return False

class AlgorithmicPolicyRunner(object):
  def __init__(self, helper, env):
    self.helper = helper
    self.env = env

  def run_episode(self):
    obs = env.reset()
    done = False
    total_reward = 0
    state = 0
    while not done:
      action = self.helper.get_action(obs, state)
      state = self.helper.get_state(obs, state)
      obs, reward, done, _ = env.step(action)
      total_reward += reward
    # Assumption (which should hold for all algorithmic envs): an episode is 
    # overall successful iff the last step has positive reward
    return reward > 0, total_reward

class BoolSatHelper(object):
  def __init__(self, solver, env, n_states):
    assert isinstance(env, gym.envs.algorithmic.algorithmic_env.AlgorithmicEnv)
    self.env = env
    # directions
    dirs, write_mode, write_chars = self.env.action_space.spaces
    assert dirs.n == 2
    self.dirs = "left", "right"
    self.n_chars = write_chars.n
    self.chars = [ chr(ord('A')+i) for i in range(write_chars.n) ]
    self.n_inputs = self.n_chars+1 # plus blank space
    self.n_outputs = self.n_chars+1 # plus "don't write"
    # All characters plus a null/blank character
    self.chars_plus = self.chars + ['_']
    self.n_states = states
    self.states = range(self.n_states)

    self.direction_rules = self._rule_vars('dir_rule', self.dirs)
    self.write_rules = self._rule_vars('write_rule', self.chars_plus)
    self.state_rules = self._rule_vars('state_rule', self.states)

    # Set of variables whose values have been used in the current session
    # Caveats:
    #   - for one-hot variables, we only need to remember whichever one was true
    #   - the last state/direction rule gets buffered. At the moment the policy
    #     fails, it's because it wrote the wrong thing (or timed out). The direction
    #     and state it would have moved to at the same step is irrelevant.
    self.dirty_variables = set()
    self.dirty_buffer = set()
    self.add_base_constraints()

  def add_base_constraints(self):
    for domain, rules in [
        (self.dirs, self.direction_rules),
        (self.chars_plus, self.write_rules),
        (self.states, self.state_rules)]:
      assert len(domain) >= 2
      if len(domain) == 2:
        # no constraints needed. law of the excluded middle, baby
        break
      else:
        for varset in rules.itervalues():
          justone = sc.ExactlyOne(*varset)
          self.solver.add_assertion(justone)


  def reset_dirty_rules(self):
    if self.dirty_variables:
      nogood = sc.And(*self.dirty_variables)
      self.solver.add_assertion(sc.Not(nogood))
    self.dirty_variables = set()
    self.dirty_buffer = set()

  def _rule_vars(self, name, to_domain):
    from_domains = [self.states, self.chars_plus]
    rules = {}
    for input_tup in itertools.product(*from_domains):
      assert len(to_domain) >= 2
      if len(to_domain) == 2:
        val = Symbol('{}_{}'.format(name, '_'.join(map(str, input_tup))), tp.BOOL)
      else:
        val = [Symbol('{}_{}__{}'.format(
                  name, '_'.join(map(str, input_tup)), output_val), tp.BOOL)
              for output_val in to_domain]
      rules[input_tup] = val
    return rules

  def get_action(self, obs, state):
    # Flush buffer, if any
    if self.dirty_buffer:
      self.dirty_variables.union(self.dirty_buffer)
      self.dirty_buffer = set()
    dirno = self._lookup(self.direction_rules, obs, state)
    writeno = self._lookup(self.write_rules, obs, state, buff=False)
    do_write = 0 if writeno == self.n_chars else 1
    to_write = min(writeno, self.n_chars-1)
    return (dirno, do_write, to_write)

  def get_state(self, obs, state):
    return self._lookup(self.state_rules, obs, state)

  def _lookup(self, rules, obs, state, buff=True):
    thing = rules[(state, obs)]
    if isinstance(thing, list):
      for (i, formula) in enumerate(thing):
        if self.solver.get_py_value(formula):
          ret = i
          form = formula
          break
    else:
      val = self.solver.get_py_value(thing)
      # Bool -> 0/1
      ret = int(val)
      # If X=0 led us astray, then put ~X in our big 'nogood' conjunction, rather than X
      form = thing if val else sc.Not(thing)

    if buff:
      self.dirty_buffer.add(form)
    else:
      self.dirty_variables.add(form)
    return ret

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
