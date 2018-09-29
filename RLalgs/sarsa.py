import numpy as np

def SARSA(env, num_episodes, gamma, lr, e):
    """Implement the SARSA algorithm following epsilon-greedy exploration.
    Update Q at the end of every episode.

    Inputs:
    env: OpenAI Gym environment
            env.P: dictionary
                    P[state][action] is tuples with (probability, nextstate, reward, terminal)
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    num_episodes: int
            Number of episodes of training
    gamma: float
            Discount factor.
    lr: float
            Learning rate.
    e: float
            Epsilon value used in the epsilon-greedy method.

    Outputs:
    Q: numpy.ndarray
            State-action values
    """
    
    ############################
    # YOUR CODE STARTS HERE

    Q = np.zeros((env.nS, env.nA))
    r_list = np.zeros(num_episodes)
    step_list = np.zeros(num_episodes)
    for i in range(num_episodes):
        s1 = 0
        a1 = ep_greedy(s1, Q, env.nA, e)
        r_list[i] += env.P[s1][a1][0][2]

        while not env.P[s1][a1][0][3]:
            s2 = env.P[s1][a1][0][1]
            a2 = ep_greedy(s1, Q, env.nA, e)
            Q[s1][a1] += lr * (env.P[s1][a1][0][2] + gamma * Q[s2][a2] - Q[s1][a1])
            s1 = s2
            a1 = a2
            r_list[i] += env.P[s1][a1][0][2]
            step_list[i] += 1
        
        if i != 0:
            r_list[i] = r_list[i - 1] + (r_list[i] - r_list[i - 1]) / (i + 1)
    # YOUR CODE ENDS HERE
    ############################

    return Q