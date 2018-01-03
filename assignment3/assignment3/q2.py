import math
import gym
from frozen_lake import *
import numpy as np
import time
import matplotlib.pyplot as plt
from utils import *


def learn_Q_QLearning(env, num_episodes=10000, gamma = 0.99, lr = 0.1, e = 0.2, max_step=6):
	"""Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy(no decay)
	Feel free to reuse your assignment1's code
	Parameters
	----------
	env: gym.core.Environment
		Environment to compute Q function for. Must have nS, nA, and P as attributes.
	num_episodes: int 
		Number of episodes of training.
	gamma: float
		Discount factor. Number in range [0, 1)
	learning_rate: float
		Learning rate. Number in range [0, 1)
	e: float
		Epsilon value used in the epsilon-greedy method. 
	max_step: Int
		max number of steps in each episode
	
	Returns
	-------
	np.array
	  An array of shape [env.nS x env.nA] representing state-action values
	"""

	Q = np.zeros((env.nS, env.nA))
	########################################################
	#                     YOUR CODE HERE                   #
	########################################################
	avg_scores = np.zeros(num_episodes)
	total_score = 0
	for episode in range(num_episodes):
		s = env.reset()
		done = False
		t = 0  # step
		while not done and t < max_step:
			# greedy
			if np.random.random() < e:
				a = np.random.randint(env.nA)
			else:
				a = np.argmax(Q[s])
			# update
			next_state, reward, done, _ = env.step(a)
			Q[s][a] += lr * (reward + gamma * np.max(Q[next_state]) - Q[s][a])
			s = next_state
			t += 1
			total_score += reward
		avg_scores[episode] = total_score / (episode + 1)
	########################################################
	#                     END YOUR CODE                    #
	########################################################
	return Q, avg_scores



def main():
	env = FrozenLakeEnv(is_slippery=False)
	print(env.__doc__)
	for e in np.linspace(0, 1, 10):
		print(e)
		Q, avg_scores = learn_Q_QLearning(env, num_episodes=10000, gamma=0.99, lr=0.1, e=e)
		render_single_Q(env, Q)
		plt.plot(avg_scores)
	plt.xlabel('episodes')
	plt.ylabel('average score')
	plt.legend(['e = ' + str(i) for i in np.linspace(0, 1, 10)], loc='upper left')
	plt.savefig('q_learning_avg_scores.png')
	plt.show()

if __name__ == '__main__':
	main()
