import numpy as np  
    
def epsilon_greedy(value, e, seed = None):
    '''
    Implement Epsilon-Greedy policy
    
    Inputs:
    value: numpy ndarray
            A vector of values of actions to choose from
    e: float
            Epsilon
    seed: None or int
            Assign an integer value to remove the randomness
    
    Outputs:
    action: int
            Index of the chosen action
    '''
    
    assert len(value.shape) == 1
    assert 0 <= e <= 1
    
    if seed != None:
        np.random.seed(seed)
    
    ############################
    # YOUR CODE STARTS HERE
    nA = value.shape[0]
    g_a = np.argmax(value) #Greedy action
    r_a = np.random.choice(nA)
    action = np.random.choice([g_a, r_a], p = [1 - e, e])
    # YOUR CODE ENDS HERE
    ############################
    return action