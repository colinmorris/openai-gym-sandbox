import gym
import time
import logging
import itertools
import sys
from pysmt.shortcuts import Symbol, Or, And
import pysmt.shortcuts as sc
import pysmt.typing as tp
from pysmt import logics


class AlgorithmicSolver(object):

  def __init__(self, env, states, solver_name):
    assert isinstance(env, gym.envs.algorithmic.algorithmic_env.AlgorithmicEnv)
    # Quantifier-free boolean logic - the simplest available, and all we need
    solver = sc.Solver(name=solver_name, logic=logics.QF_BOOL)
    self.helper = BoolSatHelper(solver, env, states)
    self.runner = AlgorithmicPolicyRunner(self.helper, env)

  def solve(self, maxiters=float('inf')):
    i = 0
    while 1 and i < maxiters:
      i += 1
      try:
        self.helper.resolve()
      except UnsatException:
        # Exhausted all possibilities
        logging.warning("Exhausted possibilities after {} iterations".format(i))
        return False
      success = self.try_model()
      if success:
        logging.info("Solved after {} iterations".format(i))
        return True
      if (i % 500) == 0:
        logging.info("i={:,}".format(i))
    # Ran out of iterations
    return False

  def try_model(self):
    max_eps = 100000 # failsafe
    i = 0
    while i < max_eps:
      i+= 1
      success, reward = self.runner.run_episode()
      if not success:
        return False
      elif reward >= env.spec.reward_threshold:
        return True
      else:
        print ".",
        # That episode was successful! We shouldn't include the rules involved
        # in it in our nogood clause
        self.helper.clear_dirty()
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

class UnsatException(Exception):
  pass

class BoolSatHelper(object):
  def __init__(self, solver, env, n_states):
    assert isinstance(env, gym.envs.algorithmic.algorithmic_env.AlgorithmicEnv)
    self.solver = solver
    self.env = env
    # directions
    dirs, write_mode, write_chars = self.env.action_space.spaces
    if dirs.n != 2:
      logging.warning("{} directions - are you sure about this?".format(dirs.n))
    self.dirs = range(dirs.n)
    self.n_chars = write_chars.n
    self.n_inputs = self.n_chars+1 # plus blank space
    self.n_outputs = self.n_chars+1 # plus "don't write"
    # All characters plus a null/blank character
    self.chars_plus = range(self.n_chars+1)
    self.n_states = n_states
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
      if len(domain) < 2:
        logging.warning("Domain with less than 2 options. Weird.")
        continue
      if len(domain) == 2:
        # no constraints needed. law of the excluded middle, baby
        continue
      else:
        for varset in rules.itervalues():
          justone = sc.ExactlyOne(*varset)
          self.solver.add_assertion(justone)


  def resolve(self):
    """We're getting read to try out another policy (because the last one failed,
    or because we're on our first iteration). Apply any new constraints we've 
    learned, then find a new solution.
    """
    if self.dirty_variables:
      nogood = sc.And(*self.dirty_variables)
      self.solver.add_assertion(sc.Not(nogood))
    self.dirty_variables = set()
    self.dirty_buffer = set()
    sat = self.solver.solve()
    if not sat:
      raise UnsatException()

  def clear_dirty(self):
    self.dirty_variables = set()
    self.dirty_buffer = set()
    
  def _rule_vars(self, name, to_domain):
    from_domains = [self.states, self.chars_plus]
    rules = {}
    assert len(to_domain) >= 1
    if len(to_domain) == 1:
      # Should only come up if max_states = 1
      logging.warning("Domain with one value: {}. Stuff may get weird.".format(name))
      val = None
    for input_tup in itertools.product(*from_domains):
      if len(to_domain) == 2:
        val = Symbol('{}_{}'.format(name, '_'.join(map(str, input_tup))), tp.BOOL)
      elif len(to_domain) > 2:
        val = [Symbol('{}_{}__{}'.format(
                  name, '_'.join(map(str, input_tup)), output_val), tp.BOOL)
              for output_val in to_domain]
      rules[input_tup] = val
    return rules

  def get_action(self, obs, state):
    # Flush buffer, if any
    if self.dirty_buffer:
      self.dirty_variables.update(self.dirty_buffer)
      self.dirty_buffer = set()
    dirno = self._lookup(self.direction_rules, obs, state)
    writeno = self._lookup(self.write_rules, obs, state, buff=False)
    do_write = 0 if writeno == self.n_chars else 1
    to_write = min(writeno, self.n_chars-1)
    return (dirno, do_write, to_write)

  def get_state(self, obs, state):
    if self.n_states == 1:
      return 0
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
        assert False, "Shouldn't have exhausted loop"
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

if __name__ == '__main__':
  try:
    env_name = sys.argv[1]
  except IndexError:
    env_name = 'Copy-v0'
    logging.warning("No environment name provided. Defaulting to {}".format(env_name))
  env = gym.make(env_name)
  t0 = time.time()
  max_states = 5
  try:
    solver_implementation = sys.argv[2]
  except IndexError:
    solver_implementation = 'z3'
  sol = AlgorithmicSolver(env, max_states, solver_implementation)
  succ = sol.solve() 
  elapsed = time.time() - t0
  print "{} after {:.1f}s".format(
    "Solved" if succ else "Exhausted policies", elapsed
  )
