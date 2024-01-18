import gymnasium as gym
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True,)
observation, info = env.reset()

# Q2.2
def random_policy(env, episodes):       # used to estimate transition and reward function
    transitions = []        # will be appended as tuple (state, action, new_state, reward)
    state_count = {}        # tracks count actions made at each state, key is tuple (state, action) and value is count or how many times that was observed
    for episode in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        while(not terminated and not truncated):
            action = env.action_space.sample()          # picks random action
            new_state, reward, terminated, truncated, _ = env.step(action)
            transitions.append([state, action, new_state, reward])
            if (state, action) in state_count:
                state_count[(state, action)] += 1
            else:
                state_count[(state, action)] = 1
            state = new_state
    print("Q2.2: Here's a list of some of our observations")
    print(transitions[1:20])
    return transitions, state_count

training_data = random_policy(env,1000) # running random policy

def next_states(state, action):     # returns the possible next states from a given state and action
    NO_MOVE_POSSIBLE = False
    future_states = []
    if action == 0:
        actual_actions = [0,1,3]
    elif action == 1:
        actual_actions = [0,1,2]
    elif action == 2:
        actual_actions = [1,2,3]
    elif action == 3:
        actual_actions = [0,2,3]

    if state in [0,1,2,3]:
        if 3 in actual_actions:
            actual_actions.remove(3)
            NO_MOVE_POSSIBLE = True
    if state in [0,4,8]:
        if 0 in actual_actions:
            actual_actions.remove(0)
            NO_MOVE_POSSIBLE = True
    if state in [13,14]:
        if 1 in actual_actions:
            actual_actions.remove(1)
            NO_MOVE_POSSIBLE = True
    if state == 3:
        if 2 in actual_actions:
            actual_actions.remove(2)
            NO_MOVE_POSSIBLE = True
    for moves in actual_actions:
        if moves == 0:
            future_states.append(state-1)
        if moves == 1:
            future_states.append(state+4)
        if moves == 2:
            future_states.append(state+1)
        if moves == 3:
            future_states.append(state-4)
    if NO_MOVE_POSSIBLE:
        future_states.append(state)

    return future_states

# Q2.3 and Q2.4
def value_iteration(transitions, state_count, gamma=0.9, num_iterations=100000):     # used to find the optimal value function
    R = {}          # will take tuple (state, action, new_state) : reward
    T = {}          # will take tuple (state, action, new_state) : probability
    policy = {}
    for transition in transitions:
        state, action, new_state, reward = transition
        if (state, action, new_state) in T:
            T[(state, action, new_state)] += 1
        else:
            T[(state, action, new_state)] = 1
        if (state, action, new_state) not in R:
            R[state, action, new_state] = reward                    # setting reward values
    for value in T:                                                 # setting transition values
        T[value] = T[value] / state_count[(value[0], value[1])]     # setting probability values

    # Extract states from transitions
    states = set(state for state, _, _, _ in transitions)
    # Initialize value function
    V = {state: 0 for state in range(16)}
    V[15] = 1
    for _ in range(num_iterations):
        for state in states:
            Q = []
            for action in [0, 1, 2, 3]:
                future_states = next_states(state, action)
                utility = 0
                for new_state in future_states:
                    utility += T[(state, action, new_state)] * (R[(state, action, new_state)] + gamma * V[new_state])   # summing all the utility for each possible new state from the given state and action
                Q.append([action, utility])
            best_utility = 0
            for item in Q:
                if item[1] >= best_utility:
                    best_utility = item[1]
                    best_move = item[0]
            V[state] = best_utility
            policy[state] = best_move           # policy extraction part
    print("Q2.3: Here is our value function after several iterations")
    print(V)
    print("Q2.4: Here is our extracted policy for each state")
    print(policy)
    return policy

# Q2.5
def act_with_policy(policy, episodes=10):       # acting with the optimal policy
    env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", render_mode = "human",is_slippery=True, )
    for episode in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        while (not terminated and not truncated):
            action = policy[state]          # picks action according to policy
            new_state, reward, terminated, truncated, _ = env.step(action)
            state = new_state
V_optimal = value_iteration(training_data[0], training_data[1])         # finding converging value function
act_with_policy(V_optimal)                                              # using the extracted policy through value iteration to act optimally
env.close()