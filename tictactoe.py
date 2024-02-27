import random 
import numpy as np

class GameState(): 
    '''
    Holds a state of the game of Tic Tac Toe. 
    The board is represented as a 3x3 numpy array.
    The player is represented as 1 or -1.
    '''
    def __init__(self, board, player):
        self.board = board
        self.player = player
        self.winner = self.get_winner() # 1, -1, 0, None 

    def simulate_action(self):
        # if opponent can win, block it
        for i in range(3):
            if np.sum(self.board[i, :]) == -2 * self.player:
                return self.take_action((i, np.where(self.board[i, :] == 0)[0][0]))
            if np.sum(self.board[:, i]) == -2 * self.player:
                return self.take_action((np.where(self.board[:, i] == 0)[0][0], i))
        if np.sum(np.diag(self.board)) == -2 * self.player:
            return self.take_action((np.where(np.diag(self.board) == 0)[0][0], np.where(np.diag(self.board) == 0)[0][0]))
        if np.sum(np.diag(np.fliplr(self.board))) == -2 * self.player:
            return self.take_action((np.where(np.diag(np.fliplr(self.board)) == 0)[0][0], np.where(np.diag(np.fliplr(self.board)) == 0)[0][0]))
        # if a win is possible, take it
        for i in range(3):
            if np.sum(self.board[i, :]) == 2 * self.player:
                return self.take_action((i, np.where(self.board[i, :] == 0)[0][0]))
            if np.sum(self.board[:, i]) == 2 * self.player:
                return self.take_action((np.where(self.board[:, i] == 0)[0][0], i))
        if np.sum(np.diag(self.board)) == 2 * self.player:
            return self.take_action((np.where(np.diag(self.board) == 0)[0][0], np.where(np.diag(self.board) == 0)[0][0]))
        if np.sum(np.diag(np.fliplr(self.board))) == 2 * self.player:
            return self.take_action((np.where(np.diag(np.fliplr(self.board)) == 0)[0][0], np.where(np.diag(np.fliplr(self.board)) == 0)[0][0]))
        
        return self.simulate_random_action()

    
    def simulate_random_action(self):
        action = random.choice(self.possible_actions())
        new_board = self.board.copy()
        new_board[action] = self.player
        new_state = GameState(new_board, -self.player)
        return new_state
    
    def take_action(self, action):
        new_board = self.board.copy()
        new_board[action] = self.player
        return GameState(new_board, -self.player)
    
    def possible_actions(self):
        return [tuple(coord) for coord in np.argwhere(self.board == 0)]
    
    def get_winner(self):
        for i in range(3):
            if abs(np.sum(self.board[i, :])) == 3 or abs(np.sum(self.board[:, i])) == 3:
                return np.sum(self.board[i, :]) // 3 or np.sum(self.board[:, i]) // 3
        if abs(np.sum(np.diag(self.board))) == 3 or abs(np.sum(np.diag(np.fliplr(self.board)))) == 3:
            return np.sum(np.diag(self.board)) // 3 or np.sum(np.diag(np.fliplr(self.board))) // 3
        if not np.any(self.board == 0):
            return 0
        
        return None
    
    def is_terminal(self): 
        if self.winner is not None:
            return True
        return False
    
    def print_board(self):
        symbols = {1: 'X', -1: 'O', 0: '-'}
        print('\n' + '\n\n'.join(['  '.join([symbols[cell] for cell in row]) for row in self.board]))