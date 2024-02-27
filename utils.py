from graphviz import Digraph
import numpy as np

def visualize_mcts_tree(root):
    dot = Digraph(comment='MCTS Tree', format='png')

    def add_nodes_edges(node):
        symbols = {1: 'X', -1: 'O', 0: '-'}
        board_str = '\n\n'.join(['  '.join([symbols[cell] for cell in row]) for row in node.state.board])
       
        label = f"{board_str}\nUCT: {node.UCT_value:.2f}\nWinner: {node.state.get_winner()}\nPlayer: {symbols[node.player]}"
        dot.node(str(id(node)), label=label, shape='box')
        
        for child in node.children:
            # Add edge from parent to child
            dot.edge(str(id(node)), str(id(child)))
            add_nodes_edges(child)
    
    def calculate_uct_value(node, C=1.4):
        if node.parent is None:
            return float('inf')  # Root node, no UCT value
        return node.value() + C * np.sqrt((2 * np.log(node.parent.visits) / node.visits))

    add_nodes_edges(root)
    return dot

dot = visualize_mcts_tree(root)
dot.render('mcts_tree_visualization', format='png', view=True)