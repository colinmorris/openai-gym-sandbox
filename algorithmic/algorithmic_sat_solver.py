"""Describe a family of deterministic policies using a bunch
of boolean variables, then use a SAT solver to do a backtracking search to find
a successful policy. Overall flow looks like:

1. ask SAT solver for a variable assignment that satisfies current constraints
2. have an agent follow the policy described by that variable assignment. If it
  wins, great. Otherwise, add a new constraint to the solver, and goto 1.

The new constraint added in 2 is the nand of all the rules that led up to our 
failure. e.g. if we see 'A', move right and write nothing, see 'B', move left and write 'C', then we add the following 'nogood' constraint:

  not( and( if_A_then_right, if_A_write_nothing, if_B_write_C ))

(In practice, the rules will be a bit more complicated than this, because most of the time the model is making decisions based not just on what it reads from the input tape, but also a 'state' variable described below.)

Solves all 1-d algorithmic environments (Copy, RepeatCopy, DuplicatedInput, Reverse), taking on the order of 10s. Haven't managed to solve addition envs yet. (Not even clear how many states are necessary. 3 is almost certainly not enough. 5 is probably enough for Reversed-Addition-v0.)
"""
import gym
import time
import logging
import numpy as np
import itertools
import sys
from pysmt.shortcuts import Symbol, Or, And
import pysmt.shortcuts as sc
import pysmt.typing as tp
from pysmt import logics
import argparse


# If we get a keyboard interrupt, continue for this many episodes, rendering them
# to stdout, before exiting. 
RENDER_BEFORE_BAIL = 5

class AlgorithmicSolver(object):

  def __init__(self, env, n_states, solver_name):
    """Except for "Copy-v0" all the algorithmic problems require some memory - 
    they can't be solved with a policy that only sees the current input character.

    States are a generic way to add the necessary expressivity. The model decides
    which way to move and which character to write based on the current input 
    character *and* the current state. It also makes a third decision at each timestep
    based on the current character/state: which state to move to next. 

    1 state (which is to say no states) is enough for Copy-v0. 2 states are enough
    for RepeatCopy, DuplicatedInput, and Reverse. Not clear how many are needed
    for the addition environments, but safe to say it's at least 3 (they use 
    ternary numbers, and the model at least needs to know which of 3 possible digits
    it's adding to the digit it's currently looking at.)
    """
    assert isinstance(env, gym.envs.algorithmic.algorithmic_env.AlgorithmicEnv)
    # Quantifier-free boolean logic - the simplest available, and all we need
    solver = sc.Solver(name=solver_name, logic=logics.QF_BOOL)
    self.helper = BoolSatHelper(solver, env, n_states)
    self.runner = AlgorithmicPolicyRunner(self.helper, env)

  def solve(self, maxiters=float('inf')):
    i = 0
    render_countdown = 0
    while 1 and i < maxiters:
      try:
        i += 1
        try:
          # Refresh our model/policy
          self.helper.resolve()
        except UnsatException:
          # Exhausted all possibilities
          logging.warning("Exhausted possibilities after {} iterations".format(i))
          return False
        # Try the new model
        success = self.try_model(render=render_countdown)
        if success:
          print "Solved after {} iterations".format(i)
          return True
        if (i % 500) == 0:
          logging.info("i={:,}".format(i))
        if render_countdown:
          render_countdown -= 1
          if render_countdown == 0:
            return False
      except KeyboardInterrupt:
        if RENDER_BEFORE_BAIL and not render_countdown:
          print "Caught keyboard interrupt. Rendering {} more episodes before exiting.".format(RENDER_BEFORE_BAIL)
          render_countdown = RENDER_BEFORE_BAIL
        else:
          raise
    # Ran out of iterations
    print "Used up all {} iterations.".format(maxiters)
    return False

  def try_model(self, render):
    max_eps = 100000 # failsafe
    i = 0
    goodeps = 0
    rewards = []
    # Run a bunch of episodes under the current policy until one episode ends
    # in failure, or we succeed enough that we reach the reward threshold.
    while i < max_eps:
      i+= 1
      success, reward = self.runner.run_episode(render)
      if not success:
        break
      else:
        goodeps += 1
        rewards.append(reward)
        if len(rewards) >= env.spec.trials and \
            np.mean(rewards[-env.spec.trials:]) >= env.spec.reward_threshold:
          return True
        # That episode was successful! We shouldn't include the rules involved
        # in it in our nogood clause
        self.helper.clear_dirty()

    if i == max_eps:
      logging.warning("Performed {} iters without failure or reaching reward\
        threshold. Sus.".format(max_eps))
    if goodeps:
      logging.info("Failed after {} successful episodes".format(goodeps))
    return False

class AlgorithmicPolicyRunner(object):
  def __init__(self, helper, env):
    self.helper = helper
    self.env = env

  def run_episode(self, render):
    obs = env.reset()
    done = False
    total_reward = 0
    state = 0
    while not done:
      action = self.helper.get_action(obs, state)
      state = self.helper.get_state(obs, state)
      obs, reward, done, _ = env.step(action)
      if render:
        env.render()
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
    """Add a priori rules.
    In practice, the only ones we know are that groups of variables representing
    a multi-valued thing should sum to 1. 
    """
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
    """Return a representation of a policy mapping (state, input_character) to 
    to_domain, some set of actions. Namely a dictionary with (state, charno)
    keys and values being a Symbol (if to_domain has two values, representable
    with one bit), or a list of symbols of length n (where n>2 is the length
    of to_domain).
    """
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
  parser = argparse.ArgumentParser()
  parser.add_argument('env', nargs='?', default='Copy-v0')
  parser.add_argument('--solver', default='msat')
  parser.add_argument('-s', '--states', type=int, default=2, 
    help='''How many states. Copy only needs 1. Other 1-d envs need 2. 
    Addition envs need more.''')
  parser.add_argument('-v', '--verbose', action='store_true')
  parser.add_argument('--monitor')
  args = parser.parse_args()
  logger = logging.getLogger()
  logger.setLevel(logging.INFO if args.verbose else logging.WARNING)
  env = gym.make(args.env)
  if args.monitor:
    env.monitor.start(args.monitor, force=True)
  t0 = time.time()
  sol = AlgorithmicSolver(env, args.states, args.solver)
  succ = sol.solve() 
  elapsed = time.time() - t0
  print "{} after {:.1f}s".format(
    "Solved" if succ else "Exhausted policies", elapsed
  )
  if args.monitor:
    env.monitor.close()
