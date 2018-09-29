import numpy as np

def QLearning(env, num_episodes, gamma, lr, e):
    """
    Implement the Q-learning algorithm following the epsilon-greedy exploration. Update Q at the end of every episode.

    Inputs:
    env: gym.core.Environment
      Environment to compute Q function
    num_episodes: int
      Number of episodes of training.
    gamma: float
      Discount factor.
    learning_rate: float
      Learning rate.
    e: float
      Epsilon value used in the epsilon-greedy method.


    Returns
    -------
    np.array
      An array of shape [env.nS x env.nA] representing state, action values
    """

    ############################
    #         YOUR CODE        #

    Q = np.zeros((env.nS, env.nA))
    r_list = np.zeros(num_episodes)
    step_list = np.zeros(num_episodes)
    for i in range(num_episodes):
        s1 = 0
        a1 = 0
        while not env.P[s1][a1][0][3]:
            a1 = ep_greedy(s1, Q, env.nA, e)
            s2 = env.P[s1][a1][0][1]
            t_q = max(Q[s1])
            Q[s1][a1] += lr * (env.P[s1][a1][0][2] + gamma * t_q - Q[s1][a1])
            s1 = s2
            r_list[i] += env.P[s1][a1][0][2]
            step_list[i] += 1
        
        if i != 0:
            r_list[i] = r_list[i - 1] + (r_list[i] - r_list[i - 1]) / (i + 1)

    ############################

    return Q