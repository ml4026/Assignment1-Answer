import numpy as np
#np.set_printoptions(precision = 3)

def value_iteration(env, gamma, max_iteration, theta):
    """
    Implement value iteration algorithm. 

    Parameters
    ----------
    env: OpenAI env.
            env.P: dictionary
                    the transition probabilities of the environment
                    P[state][action] are lists of tuples. Each tuple contains probability, nextstate, reward, terminal
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    gamma: float
            Discount factor.
    max_iteration: int
            The maximum number of iterations to run before stopping.
    theta: float
            The threshold of convergence.
    
    Outputs:
    V: numpy.ndarray
    policy: numpy.ndarray
    numIterations: int
    """

    V = np.zeros(env.nS)
    numIterations = 0

    #Implement the loop part here
    ############################
    # YOUR CODE STARTS HERE
    for i in range(max_iteration):
        delta = 0
        for s in range(env.nS):
            temp = V[s]
            max_v = -float('inf')
            for a in range(env.nA):
                models = env.P[s][a]
                t_v = 0
                for model in models:
                    prob, next_state, reward, terminal = model
                    t_v += prob * (reward + gamma * V[next_state])

                if t_v > max_v:
                    max_v = t_v

            V[s] = max_v
            delta = max(delta, abs(temp - V[s]))

        if delta < theta:
            break 
            
    numIterations = i + 1
    # YOUR CODE ENDS HERE
    ############################
    
    #Extract the "optimal" policy from the value function
    policy = extract_policy(env, V, gamma)
    
    return V, policy, numIterations

def extract_policy(env, v, gamma):

    """ Extract the optimal policy given the optimal value-function

    Inputs:
    env: OpenAI Gym environment.
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
    v: numpy.ndarray
        value function
    gamma: float
        Discount factor. Number in range [0, 1)
    
    Outputs:
    policy: numpy.ndarray
    """

    policy = np.zeros(env.nS, dtype = np.int32)
    ############################
    # YOUR CODE STARTS HERE
    for s in range(env.nS):
        argmaxa = 0
        max_v = -float('inf')
        for a in range(env.nA):
            models = env.P[s][a]
            t_v = 0
            for model in models:
                prob, next_state, reward, terminal = model
                t_v += prob * (reward + gamma * v[next_state])
                
            if t_v > max_v:
                max_v = t_v
                argmaxa = a

        policy[s] = argmaxa
    # YOUR CODE ENDS HERE
    ############################

    return policy