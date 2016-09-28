import tensorflow as tf
import numpy as np
import gym

from matplotlib import pyplot as plt

N_EPISODES = 5000
GAMMA = 0.5
RENDER_EPISODES = 0

def bias_var(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def weight_var(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def episode(sess, env, pactions, nchars, actions_size, x, render=False):
  obs = env.reset()
  last_action = [0] * actions_size
  last_char = None
  last_reward = 0
  rewards = []
  xs = []
  action_history = []
  total_reward = 0
  def pack_x():
    curr_char_arr = [0]*nchars
    curr_char_arr[obs] = 1
    last_char_arr = [0]*nchars
    if last_char is not None:
      last_char_arr[last_char] = 1
    return curr_char_arr + last_char_arr + last_action + [last_reward]

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
    last_action = actions_onehot
    if done:
      break
      
  return xs, action_history, rewards, total_reward, counter

def policy_gradient(paction_cat, actions_size):
  rewards = tf.placeholder(tf.float32, shape=[None])
  actions = tf.placeholder(tf.float32, shape=[None, actions_size])
  # These formulas are a likely source of bug. Mostly copy-pasted them without really
  # understanding them. TODO.
  focused_probs = tf.reduce_sum(tf.mul(actions, paction_cat), reduction_indices=[1])
  # What's the difference between * and tf.mul? I should know this.
  tempered = tf.mul(tf.log(focused_probs), rewards)
  loss = -tf.reduce_sum(tempered)
  optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
  return (
    rewards, actions, optimizer,
      focused_probs, tempered, loss,
  )

def discounted_rewards(r):
  r2 = np.zeros_like(r)
  running_sum = 0
  #assert r.shape[1] == 1
  for i in reversed(range(len(r))):
    running_sum = running_sum * GAMMA + r[i]
    r2[i] = running_sum 
  return r2


if __name__ == '__main__':
  # TODO: This would all have been easier if I had just used np arrays throughout rather than
  # mixing them with lists. Bleh.
  sess = tf.InteractiveSession()
  env = gym.make('Copy-v0')

  n_hidden = 50

  nchars = env.observation_space.n 
  n_actions = len(env.action_space.spaces)
  # 2 + 2 + 5 (move head, write or nah, which char to write)
  actions_size = sum(space.n for space in env.action_space.spaces)
  # Currently seen char, last seen char, last action, last reward
  input_size = nchars + nchars + actions_size + 1
  x = tf.placeholder(tf.float32, shape=[None, input_size])
  
  w1 = weight_var([input_size, n_hidden])
  b1 = bias_var([n_hidden])
  h1 = tf.nn.relu( tf.matmul(x, w1) + b1 )

  w2 = weight_var([n_hidden, actions_size])
  b2 = bias_var([actions_size])
  y = tf.nn.relu( tf.matmul(h1, w2) + b2 )
  
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
  STANDARDIZE_REWARD = 0
  env_length = 2
  for ep in range(N_EPISODES):
    xs, actions, rewards, total_reward, iters = episode(sess, env, pactions, nchars, actions_size, x)
    raw_disco_rewards = discounted_rewards(rewards)
    if STANDARDIZE_REWARD:
      disco_rewards = raw_disco_rewards - np.mean(raw_disco_rewards)
      disco_std = np.std(disco_rewards)
      if disco_std:
        disco_rewards /= np.std(disco_rewards)
    else:
      disco_rewards = raw_disco_rewards
    if DEBUG:
      pca, foc, temp, loz = sess.run([paction_cat, focprobs, tempered, loss], 
        feed_dict={x:xs, actions_ph: actions, reward_ph: disco_rewards})
      import pdb; pdb.set_trace()
    lossval, _ = sess.run([loss, optimizer], feed_dict={x: xs, actions_ph: actions, reward_ph: disco_rewards})
    if DEBUG:
      pca2 = sess.run(paction_cat, feed_dict={x: xs})
    rewards_per_ep.append(total_reward)
    iters_per_ep.append(iters)
    loss_per_ep.append(lossval)
    if env.current_length != env_length:
      print "Leveled up! Current length = {}".format(env.current_length)
      env_length = env.current_length

  for _ in range(RENDER_EPISODES):
    episode(sess, env, pactions, nchars, actions_size, x, render=True)
  #import pdb; pdb.set_trace()
  window = 200
  ravg = lambda vals: np.convolve(vals, np.ones((window,))/window, mode='valid')
  running_avg = np.convolve(rewards_per_ep, np.ones( (window,))/window, mode='valid')
  iters_running_avg = np.convolve(iters_per_ep, np.ones( (window,))/window, mode='valid')
  running_loss = ravg(loss_per_ep)
  plt.plot(running_avg)
  plt.plot(iters_running_avg, 'g--')
  plt.plot(running_loss, 'r:')
  print "Final avg reward around {:.2f}".format(running_avg[-1])
  plt.text(len(running_avg), running_avg[-1], '{:.1f}'.format(running_avg[-1]))
  plt.show()
