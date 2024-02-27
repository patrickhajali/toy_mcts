
import random 
import numpy as np
import collections

class MCTSAgent():
    def __init__(self, root):
        self.root = root

    def run_mcts(self, itermax):
        node = self.root
        while not node.is_terminal(): 
            node = self.step(node, itermax)
            
        return self.root

    def step(self, start_node, itermax):
        '''
        Runs MCTS at the given node.
        Returns the optimal action to take.
        '''
        for i in range(itermax):
            node = start_node
            node = self.selection(node)
            node.backpropagate(node.rollout())
        
        next_node = self.selection_policy(start_node)
        # next_node.UCT_value = next_node.get_UCT_value()
        return next_node
        
    def selection(self, node):
        if node.is_terminal():
            raise ValueError(f"Can't select action from terminal state {self.state}")
        
        if len(node.children) < len(node.possible_actions()): # if not all actions have been taken
            return node.expand()
        
        return self.selection_policy(node)
    
    def selection_policy(self, node, C=1):
        for child in node.children: 
            val = child.get_UCT_value(C)
            child.UCT_value = val

        vals = [child.get_UCT_value(C) for child in node.children]
        return node.children[np.argmax(vals)]
    

class MCTSNode(): 
    
    def __init__(self, state, parent=None):
        self.state = state
        self.player = self.state.player
        self.parent = parent
        self.children = []
        self.visits = 1 # changed from 0
        self.results = collections.defaultdict(int)
        self.action_counts = collections.defaultdict(int)

        self.UCT_value = float('-inf') # holds the UCT value at the time of selection

    def get_UCT_value(self, C=1.4):
        if self.parent is None:
            return float('inf')
        return self.value() + C * np.sqrt((2 * np.log(self.parent.visits) / (self.visits)))
    
    def value(self):
        return (self.results[1]  - self.results[-1])/self.visits

    def possible_actions(self):
        return self.state.possible_actions()
    
    def is_terminal(self):
        return self.state.is_terminal()
        
    def expand(self):
        '''
        Should only be called if the node is not terminal and has unexplored actions.

        Expands the node by adding a new child node for a single unexplored action.
        Returns the new child node.
        '''
        # add logging to check if being called in the right place
        actions = self.possible_actions()
        for action in actions:
            if action not in self.action_counts:
                self.action_counts[action] += 1
                opponent_state = self.state.take_action(action)
                if opponent_state.is_terminal():
                    self.children.append(MCTSNode(opponent_state, parent=self))
                    return self.children[-1]
                else:
                    new_state = opponent_state.simulate_action()
                    self.children.append(MCTSNode(new_state, parent=self))
                    return self.children[-1]


                

    def rollout(self):
        '''
        Simulate a game from the current state to a terminal state. 
        Uses a random policy to select actions for the given player and 
        an improved, heuristic-based policy for the opponent.
        '''
        curr_state = self.state
        mcts_agent = curr_state.player
        while not curr_state.is_terminal():
            curr_state = curr_state.simulate_action_and_response()

        return curr_state.get_winner()

    def backpropagate(self, result):
        '''
        How can I make this agnostic of the game type? 
        It is not agnostic because results are stored as 1, -1, 0
        which is specific to win/lose/draw.
        '''
        self.visits += 1

        if result == 0:                   # draw 
            self.results[0] += 1
        elif result == self.player: # win
            self.results[1] += 1
        else:                             # loss
            self.results[-1] += 1

        if self.parent:
            self.parent.backpropagate(result)
