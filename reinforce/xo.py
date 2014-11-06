#!/usr/bin/env python
"""
The game Tic-Tac-Toe formatting as a reinforcement learning domain.
"""
import random
import sys

import rl

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
        self.empty = range(9)
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
                    print self.pretty_board
                    print
                
                # Get player action.
                action = player.get_action(
                    state=list(self.board),
                    actions=list(self.empty))
                if action not in self.empty:
                    raise Exception, ('Player %s returned invalid action '
                        '"%s" in state "%s"') % (player, action, self.board)
                
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
                elif result == True or result == False:
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
                        print self.pretty_board
                        print
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
        