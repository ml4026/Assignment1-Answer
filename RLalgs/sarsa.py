import numpy as np
from RLalgs.utils import epsilon_greedy
import random

def SARSA(env, num_episodes, gamma, lr, e):
    """
    Implement the SARSA algorithm following epsilon-greedy exploration.

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
    #      Choose the first state of each episode randomly for exploration.
    ############################
    # YOUR CODE STARTS HERE
    env.isd = np.ones(env.nS) / env.nS
    for _ in range(num_episodes):
        state = env.reset()
        terminal = False
        action = epsilon_greedy(Q[state], e)
        while not terminal:            
            next_state, reward, terminal, prob = env.step(action)
            next_action = epsilon_greedy(Q[next_state], e)
            Q[state][action] += lr * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            state = next_state
            action = next_action
    # YOUR CODE ENDS HERE
    ############################

    return Q
