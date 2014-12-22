#!/usr/bin/env python
import random
import os
import sys
import time
import unittest

import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

import xo

class Tests(unittest.TestCase):
    
    def setUp(self):
        self.project_dir = os.path.abspath(os.path.split(__file__)[0])
        self.images_dir = os.path.join(self.project_dir, 'images')
        self.models_dir = os.path.join(self.project_dir, 'models')
    
    def test_xo(self):
    
        player1 = xo.RandomPlayer()
        series = []
        mult = 1
        games = 100000*mult
        
        # Run fixed-epsilon SARSA.
        player2 = xo.SARSAPlayer()
        player2.epsilon_decay_factor = 0
        reward_history = []
        for i in xrange(games):
            if not i % 10 or i+1 == games:
                print '\rGame %i' % i,
                sys.stdout.flush()
            game = xo.Game(players=[player1, player2])
            game.run(verbose=0)
            reward_history.append(player2.rewards[-1])
        print 
        series.append(('SARSA (fixed epsilon)', reward_history))
        
        # Run decaying-epsilon SARSA.
        player2 = xo.SARSAPlayer()
        player2.epsilon_decay_factor = 0.75 #higher=epsilon decays faster
        reward_history = []
        for i in xrange(games):
            if not i % 10 or i+1 == games:
                print '\rGame %i' % i,
                sys.stdout.flush()
            game = xo.Game(players=[player1, player2])
            game.run(verbose=0)
            reward_history.append(player2.rewards[-1])
        print 
        series.append(('SARSA (decaying epsilon %s)' \
            % player2.epsilon_decay_factor, reward_history))
        
        # Run decaying-epsilon SARSA.
        player2 = xo.SARSAPlayer()
        player2.epsilon_decay_factor = 0.25
        reward_history = []
        for i in xrange(games):
            if not i % 10 or i+1 == games:
                print '\rGame %i' % i,
                sys.stdout.flush()
            game = xo.Game(players=[player1, player2])
            game.run(verbose=0)
            reward_history.append(player2.rewards[-1])
        print 
        series.append(('SARSA (decaying epsilon %s)' \
            % player2.epsilon_decay_factor, reward_history))
#        
#        # Run decaying-epsilon SARSA.
        player2 = xo.SARSAPlayer()
        player2.epsilon_decay_factor = 0.01
        reward_history = []
        for i in xrange(games):
            if not i % 10 or i+1 == games:
                print '\rGame %i' % i,
                sys.stdout.flush()
            game = xo.Game(players=[player1, player2])
            game.run(verbose=0)
            reward_history.append(player2.rewards[-1])
        print 
        series.append(('SARSA (decaying epsilon %s)' \
            % player2.epsilon_decay_factor, reward_history))
        
        # Graph progress.
        buckets = 10*mult
        ipb = games/buckets
        for label, serie in series:
            x = np.array(xrange(buckets))
            y = [
                sum(serie[i*ipb:i*ipb+ipb])/float(ipb)
                for i in xrange(buckets)
            ]
            x_new = np.linspace(x.min(), x.max(), 300)
            print 'Creating spline...'
            y_smooth = spline(x, y, x_new)
            print 'Plotting...'
            plt.plot(x_new, y_smooth, label=label)
        legend = plt.legend(loc='best', shadow=True)
        plt.title('XO SARSA Player Against Random Player')
        fig1 = plt.gcf() # Must be before show() so we can savefig().
        plt.show()
        plt.draw()
        fig1.savefig(self.images_dir+'/sarsa-xo-progress.png', dpi=100)

    def test_xo_lfa(self):
        """
        Measure performance of linear function approximation on the XO domain.
        
        Results show it's about 5% less accurate but results in a model that's
        90 times smaller.
        """
        
        player = xo.SARSALFAPlayer()
        player.color = xo.X
        state = player.normalize_state('.x.o.....')
        self.assertEqual(state, [0, 1, 0, -1, 0, 0, 0, 0, 0])
        
        player1 = xo.RandomPlayer()
        series = []
        mult = 1
        games = 100000*mult
        
        # Run fixed-epsilon SARSA.
        player2 = xo.SARSALFAPlayer()
        player2.epsilon_decay_factor = 0 # no gradual focus on exploitation
        reward_history = []
        for i in xrange(games):
            if not i % 10 or i+1 == games:
                print '\rGame %i' % i,
                sys.stdout.flush()
            game = xo.Game(players=[player1, player2])
            game.run(verbose=0)
            reward_history.append(player2.rewards[-1])
        player2.save(self.models_dir+'/sarsa-lfa-0.dat')
        print 
        series.append(('SARSA-LFA (fixed epsilon)', reward_history))
        
        player2 = xo.SARSALFAPlayer()
        player2.epsilon_decay_factor = 0.5
        reward_history = []
        for i in xrange(games):
            if not i % 10 or i+1 == games:
                print '\rGame %i' % i,
                sys.stdout.flush()
            game = xo.Game(players=[player1, player2])
            game.run(verbose=0)
            reward_history.append(player2.rewards[-1])
        player2.save(self.models_dir+'/sarsa-lfa-05.dat')
        print 
        series.append(('SARSA-LFA (decaying epsilon %s)' \
            % player2.epsilon_decay_factor, reward_history))
        
        # Run non-LFA SARSA for comparison.
        player2 = xo.SARSAPlayer()
        player2.epsilon_decay_factor = 0.5
        reward_history = []
        for i in xrange(games):
            if not i % 10 or i+1 == games:
                print '\rGame %i' % i,
                sys.stdout.flush()
            game = xo.Game(players=[player1, player2])
            game.run(verbose=0)
            reward_history.append(player2.rewards[-1])
        player2.save(self.models_dir+'/sarsa-non-lfa-05.dat')
        print 
        series.append(('SARSA (decaying epsilon %s)' \
            % player2.epsilon_decay_factor, reward_history))
            
        # Graph progress.
        buckets = 10*mult
        ipb = games/buckets
        for label, serie in series:
            x = np.array(xrange(buckets))
            y = [
                sum(serie[i*ipb:i*ipb+ipb])/float(ipb)
                for i in xrange(buckets)
            ]
            x_new = np.linspace(x.min(), x.max(), 300)
            print 'Creating spline...'
            y_smooth = spline(x, y, x_new)
            print 'Plotting...'
            plt.plot(x_new, y_smooth, label=label)
        legend = plt.legend(loc='best', shadow=True)
        plt.title('XO SARSA-LFA Player Against Random Player')
        fig1 = plt.gcf() # Must be before show() so we can savefig().
        plt.show()
        plt.draw()
        fig1.savefig(self.images_dir+'/sarsalfa-xo-progress.png', dpi=100)

    def test_xo_lambda(self):
        """
        Measure the effect of eligibility traces in SARSA(lambda).
        """
        
        player1 = xo.RandomPlayer()
        series = []
        mult = 3
        games = int(100000*mult)
        
        player2 = xo.SARSALFAPlayer()
        player2.epsilon_decay_factor = 0.5
        player2.use_traces = True
        player2.lambda_discount = 0.5
        reward_history = []
        for i in xrange(games):
            if not i % 10 or i+1 == games:
                print '\rGame %i' % i,
                sys.stdout.flush()
            game = xo.Game(players=[player1, player2])
            game.run(verbose=0)
            reward_history.append(player2.rewards[-1])
        print
        series.append(('SARSA-LFA (lambda=%s)' % player2.lambda_discount, reward_history))
        
        player2 = xo.SARSALFAPlayer()
        player2.epsilon_decay_factor = 0.5
        player2.use_traces = True
        player2.lambda_discount = 0.5
        player2.every_step = True
        reward_history = []
        for i in xrange(games):
            if not i % 10 or i+1 == games:
                print '\rGame %i' % i,
                sys.stdout.flush()
            game = xo.Game(players=[player1, player2])
            game.run(verbose=0)
            reward_history.append(player2.rewards[-1])
        print
        series.append(('SARSA-LFA every step (lambda=%s)' % player2.lambda_discount, reward_history))
        
        player2 = xo.SARSALFAPlayer()
        player2.epsilon_decay_factor = 0.5
        player2.use_traces = True
        player2.lambda_discount = 0.9
        reward_history = []
        for i in xrange(games):
            if not i % 10 or i+1 == games:
                print '\rGame %i' % i,
                sys.stdout.flush()
            game = xo.Game(players=[player1, player2])
            game.run(verbose=0)
            reward_history.append(player2.rewards[-1])
        print
        series.append(('SARSA-LFA (lambda=%s)' % player2.lambda_discount, reward_history))
        
        player2 = xo.SARSALFAPlayer()
        player2.epsilon_decay_factor = 0.5
        player2.use_traces = True
        player2.lambda_discount = 0.9
        player2.every_step = True
        reward_history = []
        for i in xrange(games):
            if not i % 10 or i+1 == games:
                print '\rGame %i' % i,
                sys.stdout.flush()
            game = xo.Game(players=[player1, player2])
            game.run(verbose=0)
            reward_history.append(player2.rewards[-1])
        print
        series.append(('SARSA-LFA every step (lambda=%s)' % player2.lambda_discount, reward_history))
        
        player2 = xo.SARSALFAPlayer()
        player2.epsilon_decay_factor = 0.5
        player2.use_traces = False
        reward_history = []
        for i in xrange(games):
            if not i % 10 or i+1 == games:
                print '\rGame %i' % i,
                sys.stdout.flush()
            game = xo.Game(players=[player1, player2])
            game.run(verbose=0)
            reward_history.append(player2.rewards[-1])
        print
        series.append(('SARSA-LFA (non-lambda)', reward_history))
        
        # Graph progress.
        buckets = int(10*mult)
        ipb = games/buckets
        for label, serie in series:
            x = np.array(xrange(buckets))
            y = [
                sum(serie[i*ipb:i*ipb+ipb])/float(ipb)
                for i in xrange(buckets)
            ]
            x_new = np.linspace(x.min(), x.max(), 300)
            print 'Creating spline...'
            y_smooth = spline(x, y, x_new)
            print 'Plotting...'
            plt.plot(x_new, y_smooth, label=label)
        legend = plt.legend(loc='best', shadow=True)
        plt.title('XO SARSA-LFA(Lambda) Player Against Random Player')
        fig1 = plt.gcf() # Must be before show() so we can savefig().
        plt.show()
        plt.draw()
        fig1.savefig(self.images_dir+'/sarsalfa-xo-lambda-progress.png', dpi=100)

    def test_xo_fast(self):
        
        board0 = [1,2,3,4,5,6,7,8,9]

        expected_boards = [
            ('original', board0),
            ('flip-x', [3,2,1,6,5,4,9,8,7]),
            ('flip-y', [7,8,9,4,5,6,1,2,3]),
            ('rotate-90-cw', [7,4,1,8,5,2,9,6,3]),
            ('rotate-180-cw', [9,8,7,6,5,4,3,2,1]),
            ('rotate-270-cw', [3,6,9,2,5,8,1,4,7]),
            ('transpose-tl', [1,4,7,2,5,8,3,6,9]),
            ('transpose-tr', [9,6,3,8,5,2,7,4,1]),
        ]

        runs = 100000

        t0 = time.clock()
        for _ in xrange(runs):
            boards_slow = xo.transform_board(board0)
        td_slow = time.clock() - t0
        print 'slow.td:',td_slow
        print boards_slow
        self.assertEqual(len(boards_slow), 8)
        self.assertEqual(boards_slow[0], board0)
        
        for i, (name, expected_board) in enumerate(expected_boards):
            actual_board = boards_slow[i]
            #print name
            self.assertEqual(actual_board, expected_board)
        
        t0 = time.clock()
        for _ in xrange(runs):
            boards_fast = xo.xo_fast.transform_board(board0)
        td_fast = time.clock() - t0
        print 'fast.td:',td_fast
        print boards_fast
        self.assertEqual(len(boards_fast), 8)
        self.assertEqual(boards_fast[0], board0)
        
        for i, (name, expected_board) in enumerate(expected_boards):
            actual_board = boards_fast[i]
            #print name
            self.assertEqual(actual_board, expected_board)
            
        times_faster = -(td_fast - td_slow)/td_slow * 100
        print 'Cython is %.0f%% faster than Python.' % times_faster
        
        pass

    def test_flatten(self):
        b = [0, 1, 0, -1, 0, 0, 1, 0, 0]
        self.assertEqual(
            xo.expand_board(b),
            [[0, 1, 0], [-1, 0, 0], [1, 0, 0]])
        self.assertEqual(
            xo.flatten_board([[0, 1, 0], [-1, 0, 0], [1, 0, 0]]),
            b)

    def test_xo_ann(self):
        player1 = xo.RandomPlayer()
        series = []
        mult = 1
        games = int(100000*mult)
        players = []
        
        players.append(xo.ANNPlayer(
            lengths=[9, 9, 1],
            hiddenclass='TanhLayer',
            outclass='TanhLayer'))
        
        players.append(xo.ANNPlayer(
            lengths=[9, 4, 1],
            hiddenclass='TanhLayer',
            outclass='TanhLayer'))
        
        players.append(xo.ANNPlayer(
            lengths=[9, 9, 1],
            hiddenclass='TanhLayer',
            outclass='SoftmaxLayer'))
        
        players.append(xo.ANNPlayer(
            lengths=[9, 27, 1],
            hiddenclass='TanhLayer',
            outclass='TanhLayer'))
        
        players.append(xo.ANNPlayer(
            lengths=[9, 81, 1],
            hiddenclass='TanhLayer',
            outclass='TanhLayer'))
            
        for player in players:
            reward_history = []
            for i in xrange(games):
                if not i % 10 or i+1 == games:
                    print '\rGame %i of %i' % (i+1, games),
                    sys.stdout.flush()
                game = xo.Game(players=[player1, player])
                game.run(verbose=0)
                reward_history.append(player.rewards[-1] >= 0)
            print
            series.append((player.name, reward_history))
        
        # Graph progress.
        buckets = int(10*mult)
        ipb = games/buckets
        for label, serie in series:
            x = np.array(xrange(buckets))
            y = [
                sum(serie[i*ipb:i*ipb+ipb])/float(ipb)
                for i in xrange(buckets)
            ]
            x_new = np.linspace(x.min(), x.max(), 300)
            print 'Creating spline...'
            y_smooth = spline(x, y, x_new)
            print 'Plotting...'
            plt.plot(x_new, y_smooth, label=label)
        legend = plt.legend(loc='best', shadow=True)
        plt.title('XO ANN Player Against Random Player')
        fig1 = plt.gcf() # Must be before show() so we can savefig().
        plt.show()
        plt.draw()
        fig1.savefig(self.images_dir + '/ann-xo-training-progress.png', dpi=100)
        
        # Play at best.
        for player in players:
            player.best = True
            reward_history = []
            games2 = int(games*0.1)
            for i in xrange(games2):
                if not i % 10 or i+1 == games2:
                    print '\rGame %i of %i' % (i+1, games2),
                    sys.stdout.flush()
                game = xo.Game(players=[player1, player])
                game.run(verbose=0)
                reward_history.append(player.rewards[-1] >= 0)
            print
            acc = sum(reward_history)/float(len(reward_history))
            print player.name, 'accuracy: %.2f%%' % (acc*100)

if __name__ == '__main__':
    
    # Run a single test case like:
    # python test.py Tests.test_xo
    unittest.main()
    