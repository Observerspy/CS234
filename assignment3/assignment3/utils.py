import math
import gym
from frozen_lake import *
import numpy as np
import time


def render_single_Q(env, Q, max_step = 6):
	"""Renders Q function once on environment.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play Q function on. Must have nS, nA, and P as
      attributes.
    Q: np.array of shape [env.nS x env.nA]
      Q function
	"""

	state = env.reset()
	done = False
	episode_reward = 0
	count = 0
	while not done:
		# env.render()
		# time.sleep(0.5) # Seconds between frames. Modify as you wish.
		action = np.argmax(Q[state])
		state, reward, done, _ = env.step(action)
		episode_reward += reward

		count += 1
		if count >= max_step:
			break

	print("Episode reward: %d" % episode_reward)
	return episode_reward
