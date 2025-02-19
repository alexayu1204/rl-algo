from exercise1.mdp_solver import ValueIteration, PolicyIteration
from exercise1.mdp import MDP, Transition

# Create a simple MDP for testing
def create_test_mdp():
    # Define states and actions as strings (which are hashable)
    states = [str(i) for i in range(4)]  # States are '0', '1', '2', '3'
    actions = ['a0', 'a1']  # Two actions
    
    # Create MDP
    mdp = MDP()
    
    # Define transitions
    for s in states:
        for a in actions:
            # Simple transition: each action leads to the next state with probability 1
            next_state = str((int(s) + 1) % len(states))
            reward = 1.0 if next_state == '0' else 0.0
            mdp.add_transition(Transition(s, a, next_state, 1.0, reward))
    
    return mdp

def main():
    # Create MDP
    mdp = create_test_mdp()
    mdp.ensure_compiled()  # Make sure the MDP is compiled
    
    # Test Value Iteration
    print("\nTesting Value Iteration:")
    vi = ValueIteration(mdp, gamma=0.99)
    V, pi = vi.solve()
    print("Value Function:", V)
    print("Policy:", pi)
    
    # Test Policy Iteration
    print("\nTesting Policy Iteration:")
    pi_solver = PolicyIteration(mdp, gamma=0.99)
    V, pi = pi_solver.solve()
    print("Value Function:", V)
    print("Policy:", pi)

if __name__ == "__main__":
    main() 