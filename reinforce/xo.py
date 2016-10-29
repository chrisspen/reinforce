#!/usr/bin/env python
"""
The game Tic-Tac-Toe formatting as a reinforcement learning domain.
"""
from __future__ import print_function

import random
import sys

# pylint: disable=C0413

import pyximport; pyximport.install() # pylint: disable=C0321

import xo_fast # pylint: disable=E0611
from . import rl

EMPTY = '.'
X = 'x'
O = 'o'
US = 'us'
THEM = 'them'

WINNING_COMBOS = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


def transform_board(b):
    new_boards = [b]
    
    b = expand_board(b)
    
    # Flip-x.
    new_boards.append(flatten_board([list(reversed(_)) for _ in b]))
    
    # Flip-y.
    new_boards.append(flatten_board([list(_) for _ in list(reversed(b))]))
    
    #http://stackoverflow.com/q/42519/247542
    # Rotate-90.
    rot90 = [list(_) for _ in zip(*b[::-1])]
    new_boards.append(flatten_board(rot90))
    
    # Rotate-180.
    rot180 = [list(_) for _ in zip(*rot90[::-1])]
    new_boards.append(flatten_board(rot180))
    
    # Rotate-270.
    rot270 = [list(_) for _ in zip(*rot180[::-1])]
    new_boards.append(flatten_board(rot270))
    
    # Transpose from top-left to bottom-right.
    new_boards.append(flatten_board([list(_) for _ in zip(*b)]))
    
    # Transpose from top-right to bottom-left.
    new_boards.append(flatten_board([list(_) for _ in list(reversed(rot90))]))
    
    return new_boards

def flatten_board(b):
    """
    Converts a 2D list to 1D.
    """
    assert len(b) == 3 and len(b[0]) == 3
    return b[0] + b[1] + b[2]
    
def expand_board(b):
    """
    Converts a 1D list to 2D.
    """
    assert len(b) == 9
    return [b[0:3], b[3:6], b[6:9]]

class Game(rl.Domain):
    """
    Organizes a Tic-Tac-Toe match between two players.
    """
    
    def __init__(self, players=None, player1=None, player2=None, *args, **kwargs):
        if players:
            players = list(players)
            random.shuffle(players)
            self.player1 = players[0]
            self.player2 = players[1]
        else:
            assert player1 and player2
            self.player1 = player1
            self.player2 = player2
        self.player1.color = X
        self.player2.color = O
        self.board = None # board positions
        self.empty = None # indexes of empty board positions
        super(Game, self).__init__(*args, **kwargs)
        self.reset()
    
    @property
    def players(self):
        return [(X, self.player1), (O, self.player2)]
    
    def reset(self):
        self.board = [EMPTY]*9
        self.empty = list(range(9))
        self.player1.reset()
        self.player2.reset()
    
    def is_over(self):
        for combo in WINNING_COMBOS:
            colors = list(set(self.board[i] for i in combo))
            if len(colors) == 1 and colors[0] != EMPTY:
                return colors[0]
        if not self.empty:
            return True
        return False
    
    def get_color(self, player):
        for _color, _player in self.players:
            if _player == player:
                return _color
    
    def is_winner(self, player):
        result = self.is_over()
        return result == self.get_color(player)
    
    def get_actions(self, player):
        return list(self.empty)
    
    def get_other_player(self, player):
        if player == self.player1:
            return self.player2
        return self.player1
    
    @property
    def pretty_board(self):
        return '\n'.join([
            ''.join(self.board[0:3]),
            ''.join(self.board[3:6]),
            ''.join(self.board[6:9]),
        ])
    
    #TODO:support Q-value updates based on afterstate?
    #http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node68.html
    def run(self, verbose=0):
        while not self.is_over():
            for color, player in self.players:
                if verbose:
                    print(self.pretty_board)
                    print()
                
                # Get player action.
                action = player.get_action(
                    state=list(self.board),
                    actions=list(self.empty))
                if action not in self.empty:
                    raise Exception(('Player %s returned invalid action '
                        '"%s" in state "%s"') % (player, action, self.board))
                
                # Update state.
                self.board[action] = color
                self.empty.remove(action)
                
                # Check for game termination.
                result = self.is_over()
                end = bool(result)
                
                # Give player feedback.
                if result == color:
                    # Player won.
                    feedback = +1
                elif result or not result:
                    # Player drew or game ongoing.
                    feedback = 0
                else:
                    # Player lost.
                    feedback = -1
                player.reinforce(
                    feedback=feedback,
                    state=list(self.board),
                    end=end)
                
                # If it's the end of the game, give the other player feedback
                # immediately and then exit.
                if end:
                    if verbose:
                        print(self.pretty_board)
                        print()
                    self.get_other_player(player).reinforce(
                        feedback=-feedback,
                        state=list(self.board),
                        replace_last=True,
                        end=end)
                    break

class Player(rl.Agent):
    """
    An abstract agent that plays Tic-Tac-Toe.
    """
    
    def __init__(self, *args, **kwargs):
        super(Player, self).__init__(*args, **kwargs)
        self.color = None # Set by game.

    def relativize_state(self, state):
        """
        Convert literal state into relative state.
        """
        assert self.color in (X, O)
        key = {
            EMPTY: EMPTY,
            X: US if self.color == X else THEM,
            O: US if self.color == O else THEM,
        }
        state = list(key[_] for _ in state)
        return state

class RandomPlayer(Player):
    """
    Choses a random action.
    """
    
    def get_action(self, state, actions):
        """
        Retrieves the agent's action for the given state.
        """
        return random.choice(actions)

class SARSAPlayer(Player, rl.SARSAAgent):
    """
    Learns to play using basic SARSA.
    """
    
    filename = 'models/sarsa-xo-player.yaml'

    def normalize_state(self, state):
        """
        Converts state into a list of numbers that can be used in
        a linear function approximation.
        """
        state = tuple(self.relativize_state(state))
        return state

class SARSALFAPlayer(Player, rl.SARSALFAAgent):
    """
    Learns to play using basic SARSA and linear function approximation.
    """
    
    filename = 'models/sarsalfa-xo-player.yaml'
    
    LFA_KEY = {
        EMPTY: 0,
        US: 1,
        THEM: -1,
    }

    def normalize_state(self, state):
        """
        Converts state into a list of numbers that can be used in
        a linear function approximation.
        """
        state = self.relativize_state(state)
        state = [self.LFA_KEY[_] for _ in state]
        return state

class ANNPlayer(Player, rl.ANNAgent):
    """
    Learns to play using an artificial neural network.
    
    Works by using the ANN to estimate the the expected reward after
    performing each legal action and recommends the action corresponding
    to the highest expected reward. 
    """
    
    filename = 'models/ann-xo-player.yaml'
    
    symbol_to_int = {
        US: +1,
        EMPTY: 0,
        THEM: -1,
    }

    def __init__(self, *args, **kwargs):
        Player.__init__(self, *args, **kwargs)
        rl.ANNAgent.__init__(self, *args, **kwargs)

    def normalize_state(self, state):
        """
        Converts state into a list of numbers.
        """
        assert self.color is not None
        
        # Make the board relative to us.
        if not (US in state or THEM in state):
            state = self.relativize_state(state)
        
        # Convert to integers suitable for input into the ANN.
        state = [self.symbol_to_int[_] for _ in state]
        
        # Convert one of 8 possible symmetric versions to the standard.
        boards = sorted(xo_fast.transform_board(state))
        state = boards[0]
        
        return state

    def simulate_action(self, state, action):
        """
        Returns the expected next-state if the given action is performed
        in the given state.
        """
#        print 'state0:',state,action
        state = self.relativize_state(state)
        invalid = set(state).difference([US, THEM, EMPTY])
        assert not invalid, invalid
        assert state[action] == EMPTY
        state = list(state)
        state[action] = US
#        print 'state1:',state
        return state
