import numpy as np
from RLalgs.policy_utils import epsilon_greedy
import random

def SARSA(env, num_episodes, gamma, lr, e):
    """
    Implement the SARSA algorithm following epsilon-greedy exploration. Update Q at the end of every episode.

    Inputs:
    env: OpenAI Gym environment 
            env.P: dictionary
                    P[state][action] are tuples of tuples tuples with (probability, nextstate, reward, terminal)
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
    
    Q = np.zeros((env.nS, env.nA))
    
    #TIPS: Call function epsilon_greedy without setting the seed
    ############################
    # YOUR CODE STARTS HERE
    for i in range(num_episodes):
        s1 = random.choice(range(env.nS))
        a1 = epsilon_greedy(Q[s1], e)
        terminal = False
        
        while not terminal:
            models = env.P[s1][a1]
            model = random.choice(models)
            prob, s2, reward, terminal = model
            a2 = epsilon_greedy(Q[s2], e)
            Q[s1][a1] += lr * (reward + gamma * Q[s2][a2] - Q[s1][a1])
            s1 = s2
            a1 = a2
    # YOUR CODE ENDS HERE
    ############################

    return Q