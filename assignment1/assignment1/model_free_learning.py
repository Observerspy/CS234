# Episode model free learning using Q-learning and SARSA

# Do not change the arguments and output types of any of the functions provided! You may debug in Main and elsewhere.

import numpy as np
import gym
import time
from lake_envs import *
import matplotlib.pyplot as plt

def learn_Q_QLearning(env, num_episodes=2000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
    """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
    Update Q at the end of every episode.
    
    Parameters
    ----------
    env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
    num_episodes: int 
    Number of episodes of training.
    gamma: float
    Discount factor. Number in range [0, 1)
    learning_rate: float
    Learning rate. Number in range [0, 1)
    e: float
    Epsilon value used in the epsilon-greedy method. 
    decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)
    
    Returns
    -------
    np.array
    An array of shape [env.nS x env.nA] representing state, action values
    """

    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    Q = np.zeros((env.nS, env.nA))
    for i in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            # greedy
            if np.random.random() < e:
                a = np.random.randint(env.nA)
            else:
                a = np.argmax(Q[s])
            # update
            next_state, reward, done, _ = env.step(a)
            Q[s][a] += lr * (reward + gamma * np.max(Q[next_state]) - Q[s][a])
            s = next_state
        if i % 10 == 0:
            e *= decay_rate
    return Q

def learn_Q_SARSA(env, num_episodes=2000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
    """Learn state-action values using the SARSA algorithm with epsilon-greedy exploration strategy
    Update Q at the end of every episode.
    
    Parameters
    ----------
    env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
    num_episodes: int 
    Number of episodes of training.
    gamma: float
    Discount factor. Number in range [0, 1)
    learning_rate: float
    Learning rate. Number in range [0, 1)
    e: float
    Epsilon value used in the epsilon-greedy method. 
    decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)
    
    Returns
    -------
    np.array
    An array of shape [env.nS x env.nA] representing state-action values
    """

    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    Q = np.zeros((env.nS, env.nA))
    for i in range(num_episodes):
        s = env.reset()
        done = False
        # greedy
        if np.random.random() < e:
            a = np.random.randint(env.nA)
        else:
            a = np.argmax(Q[s])
        while not done:
            # update
            next_state, reward, done, _ = env.step(a)
            # greedy
            if np.random.random() < e:
                next_a = np.random.randint(env.nA)
            else:
                next_a = np.argmax(Q[next_state])
            Q[s][a] += lr * (reward + gamma * Q[next_state][next_a] - Q[s][a])
            s = next_state
            a = next_a
        if i % 10 == 0:
            e *= decay_rate
    return Q


def render_single_Q(env, Q):
    """Renders Q function once on environment. Watch your agent play!
    
    Parameters
    ----------
    env: gym.core.Environment
      Environment to play Q function on. Must have nS, nA, and P as
      attributes.
    Q: np.array of shape [env.nS x env.nA]
      state-action values.
    """

    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        # env.render()
        # time.sleep(0.5) # Seconds between frames. Modify as you wish.
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        episode_reward += reward

    print("Episode reward: %f" % episode_reward)
    return episode_reward


# Feel free to run your own debug code in main!
def main():
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    Q = learn_Q_QLearning(env, num_episodes=1000)
    # Q = learn_Q_SARSA(env)
    score = []
    for i in range(100):
        episode_reward = render_single_Q(env, Q)
        score.append(episode_reward)
    for i in range(len(score)):
        score[i] = np.mean(score[:i + 1])
    plt.plot(np.arange(100), np.array(score))
    plt.title('The running average score of the Q-learning agent')
    plt.xlabel('traning episodes')
    plt.ylabel('score')
    plt.savefig('q_learning.png')
    plt.show()

if __name__ == '__main__':
    main()
