import math
import time
import random
from sample_players import DataPlayer

PLAYER_1 = 1
NON_VISITED, VISITED = 0,1
UCB_CONSTANT = math.sqrt(2)

def get_state_hash(state):
    # used for debegging
    locs_hash = hash(state.locs) if None not in state.locs else hash((None, None))
    combined_hash = hash((state.board, state.ply_count, locs_hash))   
    return combined_hash

class MCTSNode:
    def __init__(self, state, parent=None, action=None, context=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 1
        self.untried_actions = self.state.actions()  
        self.context = context or {}

    def uct_tree_policy(self):
        current_node = self
        
        while not current_node.state.terminal_test(): 
            if not current_node.is_fully_expanded():
                current_node.expand()
            else:
                current_node = current_node.uct_best_child()

        return current_node
    
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.result(action) 
        child_node = MCTSNode(next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node
    
    def uct_best_child(self, c= UCB_CONSTANT): 
        choices_weights = [
            (child.wins / child.visits) + c * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]
  

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """

    def __init__(self, player_id =PLAYER_1, scoring_fn=None):
        super().__init__(player_id =PLAYER_1)
        self.scoring_fn = scoring_fn
        self.player_id = player_id
        self.context = {}

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        
        if state.ply_count < 2 or state.terminal_test():
            self.queue.put(random.choice(state.actions()))
        else:
            self.root = MCTSNode(state, context=self.context) 
            self.uct_search()
            self.queue.put(self.root.uct_best_child(0).action)
           
    def uct_default_policy(self,state):
            """
            Simulate a random play-out from the given state and return the result.
            """
            current_state = state
            while not current_state.terminal_test():
                action = random.choice(current_state.actions())
                current_state = current_state.result(action)
            return  1 if current_state.utility(self.player_id) > 0 else -1
        
    def uct_search(self):
        def uct_backup_negamax(mcts_node, reward):
            """
            Apply the negamax update rule for backpropagation.
            """
            current_node = mcts_node
            while current_node is not None:
                current_node.visits += 1
                current_node.wins += reward
                # Negate the reward for the opponent.
                reward = -reward
                current_node = current_node.parent

        leaf_node = self.root.uct_tree_policy()
        # Simulate
        reward = self.uct_default_policy(leaf_node.state)
        # Backpropagate
        uct_backup_negamax(leaf_node,reward)
