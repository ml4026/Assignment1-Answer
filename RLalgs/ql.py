import numpy as np
from RLalgs.utils import epsilon_greedy
import random

def QLearning(env, num_episodes, gamma, lr, e):
    """
    Implement the Q-learning algorithm following the epsilon-greedy exploration. Update Q at the end of every episode.

    Inputs:
    

    Outputs:
    Q: numpy.ndarray
    """

    Q = np.zeros((env.nS, env.nA))
    
    ############################
    # YOUR CODE STARTS HERE
    for i in range(num_episodes):
        s1 = random.choice(range(env.nS))
        terminal = False
        
        while not terminal:
            a1 = epsilon_greedy(Q[s1], e)
            models = env.P[s1][a1]            
            model = random.choice(models)
            prob, s2, reward, terminal = model
            Q[s1][a1] += lr * (reward + gamma * np.amax(Q[s2]) - Q[s1][a1])
            s1 = s2
    # YOUR CODE ENDS HERE
    ############################

    return Q