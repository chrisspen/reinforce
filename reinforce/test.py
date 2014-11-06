#!/usr/bin/env python
import random
import sys
import unittest

import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

import xo

class Tests(unittest.TestCase):
    
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
        fig1.savefig('images/sarsa-xo-progress.png', dpi=100)

    def test_xo_lfa(self):
        """
        Measure performance of linear function approximation on the XO domain.
        
        Results show it's about 5% less accurate but results in a model that's
        90 times smaller.
        """
        
        player = xo.SARSALFAPlayer()
        player.color = xo.X
        state = player.normalize_state('.x.o....')
        self.assertEqual(state, [0, 1, 0, -1, 0, 0, 0, 0])
        
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
        player2.save('models/sarsa-lfa-0.dat')
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
        player2.save('models/sarsa-lfa-05.dat')
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
        player2.save('models/sarsa-non-lfa-05.dat')
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
        fig1.savefig('images/sarsalfa-xo-progress.png', dpi=100)

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
        fig1.savefig('images/sarsalfa-xo-lambda-progress.png', dpi=100)


if __name__ == '__main__':
    
    # Run a single test case like:
    # python test.py Tests.test_xo
    unittest.main()
    