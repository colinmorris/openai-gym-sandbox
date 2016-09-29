from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import gym
import sys

#ENV_NAME = 'Copy-v0'
ENV_NAME = 'DuplicatedInput-v0'

_CONFIGS = {
  'Copy-v0': dict(
    gamma=.5,
    zero_penalty=-.6
  ),
  'DuplicatedInput-v0': dict(
    zero_penalty=-.001,
    input_last_char=1,
    input_last_action=1,
    learning_rate=0.01,
  ),
}
  

_CONFIG = _CONFIGS[ENV_NAME]

N_EPISODES = 10000
EARLY_EXIT = 1
# Does setting this to high values encourage dilly-dallying? (And does ZERO_PENALTY offset that?)
GAMMA = _CONFIG.get('gamma', .9)
RENDER_EPISODES = 0
L2 = 1
L2_WEIGHT = 0.001
COMPLETION_BONUS = 0
LEARNING_RATE = _CONFIG.get('learning_rate', 0.001)

INPUT_LAST_CHAR = _CONFIG.get('input_last_char', 0)
INPUT_LAST_ACTION = _CONFIG.get('input_last_action', 0)
INPUT_LAST_REWARD = 0

N_HIDDEN = 40

TELESCOPING_REWARDS = 0

# This hurts things very badly. iunno why. Got the idea from here:
#     https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
# Maybe it makes more sense for that kind of problem where one episode 
# can have thousands of transitions?
STANDARDIZE_REWARD = 0

# Setting this to -.2 or less seems to help a lot for copy (almost necessary)
# Sweet spot maybe around -.6
# Obviously a high value of this is not good for dedupe, but it can tolerate 
# a penalty <= -.3. A very small penalty like -.01 might even help? Haven't
# collected a ton of data though.
ZERO_PENALTY = _CONFIG.get('zero_penalty', 0)

def bias_var(shape):
  initial = tf.constant(.1, shape=shape)
  return tf.Variable(initial)

def weight_var(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def episode(sess, env, pactions, nchars, actions_size, x, render=False):
  obs = env.reset()
  # Last position represents no previous action taken
  last_action = [0] * actions_size + [1]
  last_char = None
  last_reward = 0
  rewards = []
  xs = []
  action_history = []
  total_reward = 0
  def pack_x():
    xarr = [0]*nchars
    xarr[obs] = 1
    if INPUT_LAST_CHAR:
      last_char_arr = [0]*nchars
      if last_char is not None:
        last_char_arr[last_char] = 1
      xarr += last_char_arr
    if INPUT_LAST_ACTION:
      xarr += last_action
    if INPUT_LAST_REWARD:
      xarr += [last_reward]
    return xarr

  counter = 0
  for i in range(100):
    counter += 1
    if render:
      env.render()
    curr_x = pack_x()
    xs.append(curr_x)
    action_probs = sess.run(pactions, feed_dict={x: [curr_x]})
    # choose next action
    actions = []
    actions_onehot = [0] * actions_size
    onehot_offset = 0
    for i, probs in enumerate(action_probs):
      assert probs.shape[0] == 1
      probs = probs[0]
      sample = np.sum(np.cumsum(probs) < np.random.rand())
      actions_onehot[onehot_offset+sample] = 1
      onehot_offset += probs.shape[0]
      actions.append(sample)

    action_history.append(actions_onehot)
    assert env.action_space.contains(actions)
    last_char = obs
    obs, last_reward, done, _ = env.step(actions)
    total_reward += last_reward
    rewards.append(last_reward)
    last_action = actions_onehot + [0]
    if done:
      if render:
        env.render()
      break
      
  return xs, action_history, rewards, total_reward, counter

def policy_gradient(paction_cat, actions_size):
  rewards = tf.placeholder(tf.float32, shape=[None])
  actions = tf.placeholder(tf.float32, shape=[None, actions_size])
  # For each transition, isolate the probabilities of the actions that
  # were actually performed.
  focused_probs = tf.reduce_sum(tf.mul(actions, paction_cat), reduction_indices=[1])
  # Starting from gradients that encourage the performed actions, multiply by
  # the 'advantage'. If the performed actions were bad, this should be a negative
  # multiplier, thus discouraging that set of actions.
  # (The particular loss function I'm using here is in imitation of this example:
  #     https://github.com/kvfrans/openai-cartpole/blob/master/cartpole-policygradient.py
  #  not sure if there are other reasonable ways to do it.)
  
  tempered = tf.mul(tf.log(focused_probs), rewards)
  #tempered = tf.mul(focused_probs, rewards)
  loss = -tf.reduce_sum(tempered)
  if L2:
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    loss = loss + L2_WEIGHT * l2_loss
  # I have no idea what I'm doing
  optimizer = tf.train.AdamOptimizer(LEARNING_RATE, beta1=.8, beta2=.99).minimize(loss)
  #optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
  #optimizer = tf.train.AdadeltaOptimizer(LEARNING_RATE).minimize(loss)
  return (
    rewards, actions, optimizer,
      focused_probs, tempered, loss,
  )

def discounted_rewards(r):
  if TELESCOPING_REWARDS:
    running = 1
    for i, ree in enumerate(r):
      if ree == 1:
        r[i] = running
        running += 1
        #running = running*2 + 4
  r2 = np.zeros_like(r)
  running_sum = 0
  bonus = COMPLETION_BONUS if (COMPLETION_BONUS and r[-1] == 1) else 1
  for i in reversed(range(len(r))):
    # The min here is basically cheating :/
    #running_sum = min(running_sum * GAMMA,0) + r[i]
    rew = r[i]
    # Also cheating
    if ZERO_PENALTY and r[i] == 0:
      rew = ZERO_PENALTY
    running_sum = running_sum * GAMMA + rew
    r2[i] = running_sum * bonus
  return r2


if __name__ == '__main__':
  # TODO: This would all have been easier if I had just used np arrays throughout rather than
  # mixing them with lists. Bleh.
  sess = tf.InteractiveSession()
  env = gym.make(ENV_NAME)
  ''' experiment
  gym.envs.register(
      id='Reverse-v1',
      entry_point='gym.envs.algorithmic:ReverseEnv',
      kwargs={'base': 5},
      timestep_limit=200,
      reward_threshold=25.0,
  )
  env = gym.make('Reverse-v1')
  '''
  nchars = env.observation_space.n 
  n_actions = len(env.action_space.spaces)
  # 2 + 2 + 5 (move head, write or nah, which char to write)
  actions_size = sum(space.n for space in env.action_space.spaces)
  # Currently seen char, last seen char, last action, last reward
  input_size = (nchars + 
    (INPUT_LAST_CHAR * nchars) + 
    # We'll add one more input which represents "no action taken" (i.e. this is the first step)
    INPUT_LAST_ACTION * (actions_size+1) + 
    INPUT_LAST_REWARD * 1
  )
  x = tf.placeholder(tf.float32, shape=[None, input_size])
  w1 = weight_var([input_size, N_HIDDEN])
  b1 = bias_var([N_HIDDEN])
  h1 = tf.nn.relu( tf.matmul(x, w1) + b1 )
  w2 = weight_var([N_HIDDEN, actions_size])
  b2 = bias_var([actions_size])
  y = tf.nn.relu( tf.matmul(h1, w2) + b2 )
  
  # Probabilities of each action
  pactions = []
  offset = 0
  for space in env.action_space.spaces:
    slice_ = tf.slice(y, [0, offset], [-1, space.n])
    paction = tf.nn.softmax(slice_)
    pactions.append(paction)
    offset += space.n
  paction_cat = tf.concat(1, pactions)

  reward_ph, actions_ph, optimizer,\
    focprobs, tempered, loss = policy_gradient(paction_cat, actions_size)
  
  sess.run(tf.initialize_all_variables())
  
  rewards_per_ep = []
  iters_per_ep = []
  loss_per_ep = []
  DEBUG = 0
  env_length = env.current_length

  # Experimented with batching. Seemed to hurt more than help.
  EPISODES_PER_BATCH = 1
  # Experiment: give all actions in an episode the same score (1,-1) according to
  # whether the episode ended successfully. Didn't really work out that well.
  REWARD_EXPERIMENT = 0
  xs = []
  actions = []
  rewards = []
  successes = []
  for ep in range(N_EPISODES):
    epxs, epactions, eprewards, total_reward, iters = episode(sess, env, pactions, nchars, actions_size, x)
    if REWARD_EXPERIMENT:
      disco_rewards = [total_reward for _ in eprewards]
    else:
      disco_rewards = discounted_rewards(eprewards)

    successes.append(1 if eprewards[-1] == 1 else 0)
    xs += epxs
    actions += epactions
    rewards = np.concatenate([rewards, disco_rewards])
    if (ep+1) % EPISODES_PER_BATCH == 0:
      if DEBUG and ep > 4000:
        import pdb; pdb.set_trace()
        wut, but, why = sess.run([w1, b1, y], feed_dict={x:xs})
      if STANDARDIZE_REWARD:
        rewards = rewards - np.mean(rewards)
        std = np.std(rewards)
        if std:
          rewards /= std
      
      if DEBUG: 
        pca, foc, temp, loz = sess.run([paction_cat, focprobs, tempered, loss], 
          feed_dict={x:xs, actions_ph: actions, reward_ph: rewards})
        import pdb; pdb.set_trace()
      
      lossval, _ = sess.run([loss, optimizer], feed_dict={x: xs, actions_ph: actions, reward_ph: rewards})
      loss_per_ep.append(lossval)
      
      if DEBUG:
        pca2 = sess.run(paction_cat, feed_dict={x: xs})
      
      xs = []
      actions = []
      rewards = []

    rewards_per_ep.append(total_reward)
    iters_per_ep.append(iters)

    # Have we graduated to the next curriculum?
    if env.current_length != env_length:
      sys.stderr.write("Leveled up! Current length = {} (ep={})\n".format(
        env.current_length, ep))
      env_length = env.current_length
      if env.current_length == 30 and EARLY_EXIT:
        break

  if RENDER_EPISODES:
    env.monitor.start('/tmp/dedupe-1', force=True)
    for i in range(RENDER_EPISODES):
      episode(sess, env, pactions, nchars, actions_size, x, render=True)
      if i != RENDER_EPISODES-1:
        print
        print "*" * 80
        print
    env.monitor.close()
  window = 200 + N_EPISODES/100
  ravg = lambda vals: np.convolve(vals, np.ones((window,))/window, mode='valid')
  running_avg = np.convolve(rewards_per_ep, np.ones( (window,))/window, mode='valid')
  iters_running_avg = np.convolve(iters_per_ep, np.ones( (window,))/window, mode='valid')
  running_loss = ravg(loss_per_ep)
  running_successes = ravg(successes)
  plt.plot(running_avg)
  plt.plot(iters_running_avg, 'g--')
  plt.plot(running_successes, 'c:')
  #plt.plot(running_loss, 'r:')
  print "Final avg reward around {:.2f}".format(running_avg[-1])
  plt.text(len(running_avg), running_avg[-1], '{:.1f}'.format(running_avg[-1]))

  if len(sys.argv) > 1:
    fname = sys.argv[1]
    print "Saving plot to {}".format(fname)
    plt.savefig(fname)
  else:
    plt.show()
