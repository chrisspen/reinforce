# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import random

import six
from six.moves import range
import yaml

from pybrain.tools.shortcuts import buildNetwork
from pybrain import structure as pybrain_structure
#import TanhLayer, SoftmaxLayer, SigmoidLayer, GaussianLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

class Distribution(object):
    """
    Tracks statistics for a stream of discrete events.
    """
    
    def __init__(self):
        self._d = {}
        self._total = 0
        
    def __iadd__(self, v):
        self._d.setdefault(v, 0)
        self._d[v] += 1
        self._total += 1
        return self
        
    @property
    def total(self):
        return self._total
        
    @property
    def median(self):
        if not self._d:
            return
        values = sorted((_b, _a) for _a, _b in self._d.iteritems())
        return values[-1][1]
        
    @property
    def histogram(self):
        return self._d.copy()
    
    @property
    def nhistogram(self):
        return dict((_a, _b/float(self._total)) for _a, _b in self._d.iteritems())
        
class Aggregate(object):
    """
    Tracks statistics for a stream of continuous events.
    """
    
    def __init__(self):
        self._sum = 0.
        self._total = 0
        self._values = []
        self._min = 1e9999999
        self._max = -1e9999999
        
    def __iadd__(self, v):
        v = float(v)
        self._values.append(v)
        self._sum += v
        self._total += 1
        self._min = min(self._min, v)
        self._max = max(self._max, v)
        return self
    
    @property
    def total(self):
        return self._total
        
    @property
    def max(self):
        return self._max
    
    @property
    def min(self):
        return self._min
    
    @property
    def median(self):
        if not self._values:
            return
        values = sorted(self._values)
        return values[int(len(values)/2.)]
        
    @property
    def mean(self):
        if not self._total:
            return
        return self._sum / self._total

class ArgMaxSet(object):
    """
    Tracks one or more objects corresponding to a maximum score.
    
    Similar in concept to the traditional argmax() but tracks objects that have
    the same score and randomly choices between them for the final selection.
    """
    
    def __init__(self):
        self._score = -1e9999999999
        self._objects = set()
        
    def add(self, score, obj):
        if score > self._score:
            self._score = score
            self._objects = set([obj])
        elif score == self._score:
            self._objects.add(obj)
    
    @property
    def score(self):
        return self._score
    
    @property    
    def obj(self):
        if not self._objects:
            return
        return random.choice(list(self._objects))

def weighted_choice(choices, get_total=None, get_weight=None):
    """
    A version of random.choice() that accepts weights attached to each
    item, increasing or decreasing the likelyhood that each will be picked.
    
    Paramters:
        
        choices := can be either:
            1. a list of the form `[(item, weight)]`
            2. a dictionary of the form `{item: weight}`
            3. a generator that yields `(item, weight)`
    
        get_total := In some cases with large numbers of items, it may be more
            efficient to track the `total` separately and pass it in at call
            time, and then pass in a custom iterator that lazily looks up the
            item's weight. Depending on your distribution, this should
            consume much less memory than loading all items immediately.
    
    Note, this assumes all weights are >= 0.0.
    Negative weights may throw an exception.
    
    """
    
    def get_iter():
        if isinstance(choices, dict):
            return choices.iteritems()
        return choices
            
    if callable(get_total):
        total = get_total()
    else:
        total = sum(w for c, w in get_iter())
    
    # If no non-zero weights given, then just use a uniform distribution.
    if not total:
        return random.choice(list(get_iter()))[0]
        
    r = random.uniform(0, total)
    upto = 0.
    for c in get_iter():
        if get_weight:
            w = get_weight(c)
        else:
            c, w = c
        if w < 0:
            raise Exception('Invalid negative weight: %s' % (w,))
        if upto + w >= r:
            return c
        upto += w
    raise Exception('Unable to make weighted choice: total=%s, choices=%s' % (total, choices))

class Domain(object):
    """
    A particular type of problem which provides an explicit reward.
    """
    
    def __init__(self):
        pass

    def get_actions(self, agent):
        """
        If supported, returns a list of legal actions for the agent
        in the current state.
        """
        raise NotImplementedError

    def run(self, fn):
        """
        Runs the interactive learning process.
        """
        raise NotImplementedError
        
class Agent(object):
    """
    The thing that attempts to learn in order to accomplish a goal.
    """
    
    filename = 'models/agent.yaml'
    
    def __init__(self):
        self.best = False
    
    def reset(self):
        """
        Clears episodic variables.
        """
        
    def get_action(self, state, actions):
        """
        Retrieves the agent's action for the given state.
        """
        raise NotImplementedError
    
    def reinforce(self, feedback, state=None, replace_last=False, end=None):
        """
        Processes a feedback signal (e.g. reward or punishment).
        
        If supported by the domain, end represents the end of the episode.
        
        If the state changed since the last time the agent viewed it, state
        will be the current state.
        """

    def record_history(self, tpl):
        if self.history and isinstance(tpl, float):
            assert not isinstance(self.history[-1], float), \
                'Double feedback append.'
        self.history.append(tpl)
        
    @classmethod
    def load(cls, fn=None, ignore_errors=False):
        fn = fn or cls.filename
        player = cls()
        d = {}
        try:
            d = yaml.load(open(fn))
        except Exception as e:
            if ignore_errors:
                print(e)
        player.__setstate__(d)
        return player
    
    def save(self, fout=None):
        fout = fout or self.filename
        if isinstance(fout, six.string_types):
            fout = open(fout, 'w')
        yaml.dump(self.__getstate__(), fout)

class SARSAAgent(Agent):
    """
    Uses basic SARSA to learn.
    
    (gamma) - reward discount factor (between 0 and 1)
    (alpha) - learning rate (between 0 and 1)
    (epsilon) - parameter for the epsilon-greedy policy (between 0 and 1)
    (lambda) - parameter for the SARSA(lambda) learning algorithm 
    """
    
    def __init__(self, alpha=0.1, gamma=1.0, epsilon=0.1, use_traces=False, *args, **kwargs):
        super(SARSAAgent, self).__init__(*args, **kwargs)
        self.Q = {} # {s:{a:q-value}} default 0
        self.alpha = alpha
        self.gamma = gamma
        self._epsilon = self.epsilon = epsilon
        self.epsilon_decay_factor = 0.5
        self.episodes = 0
        
        # If true, uses eligibility traces to update Q.
        # http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node73.html
        # http://en.wikipedia.org/wiki/Gamma
        self.use_traces = use_traces
        
        #TODO:decay epsilon over episodes?
        self.reset()
    
    def __setstate__(self, d=None):
        d = d or {}
        self.__dict__.update(d)
        
    def __getstate__(self):
        return dict(
            Q=self.Q,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=self.epsilon,
            episodes=self.episodes,
            use_traces=self.use_traces,
        )
        
    def reset(self):
        """
        Clears episodic variables.
        """
        self.history = []
        self.rewards = []
        self._epsilon = \
            self.epsilon/(1+self.episodes*self.epsilon_decay_factor)
        self.episodes += 1
    
    def normalize_state(self, state):
        return tuple(state)
    
    def get_Q(self, state, action):
        state = tuple(state)
        self.Q.setdefault(state, {})
        self.Q[state].setdefault(action, 0.)
        return self.Q[state][action]
    
    def set_Q(self, state, action, value):
        state = tuple(state)
        self.Q.setdefault(state, {})
        self.Q[state].setdefault(action, 0.)
        self.Q[state][action] = value
    
    def get_action(self, state, actions):
        """
        Retrieves the agent's action for the given state.
        """
        
        state = self.normalize_state(state)
        
        # Select action according to an epsilon-greedy strategy.
        if random.random() > self._epsilon:
            # Exploit our Q-values.
            best = (-1e999999999999, None)
#            print 'exploit'
            for action in actions:
                best = max(best, (self.get_Q(state, action), action))
            action = best[1]
        else:
            # Otherwise explore something random.
#            print 'explore'
            action = random.choice(actions)
        
        # Record (state,action) tuple for later learning update.
        self.record_history((state, action))
        
        return action
    
    def reinforce(self, feedback, state=None, replace_last=False, end=None):
        """
        Processes a feedback signal (e.g. reward or punishment).
        """
        Q = self.get_Q
        
        if state:
            state = self.normalize_state(state)
        
        # Record feedback.
        if replace_last:
            # Overwrite our last reward, since it was incorrect as our
            # opponent did something to effect it in hindsight.
            assert not isinstance(self.history[-1], tuple), self.history
            self.history[-1] = feedback
            self.rewards[-1] = feedback
        else:
            if self.history:
                assert not isinstance(self.history[-1], float), 'Double feedback append.'
            self.history.append(feedback)
            self.rewards.append(feedback)
            
        # Insert the null state-action pair to terminate the episode.
        if end:
            assert state is not None
            self.history.append((state, None))
        
        # Update Q-values using the SARSA algorithm.
        # http://en.wikipedia.org/wiki/SARSA
        if len(self.history) >= 3:
            #print 'self.history:',self.history
            if end:
                (s0, a0), r1, (s1, a1) = self.history[-3:]
            else:
                (s0, a0), r1, (s1, a1) = self.history[-4:-1]
            self.set_Q(s0, a0, Q(s0, a0) + self.alpha*(r1 + self.gamma*Q(s1, a1) - Q(s0, a0)))

EVERY_STEP = 'every-step'
END_OF_EPISODE = 'end-of-episode'

REINFORCE_METHODS = (
    EVERY_STEP,
    END_OF_EPISODE,
)

class SARSALFAAgent(Agent):
    
    """
    Uses linear function approximation of features in the state to learn
    Q-values instead of trying to index all possible states.
    
    http://www.scholarpedia.org/article/Temporal_difference_learning#TD_with_Function_Approximation
    
    Pros:
    For large domains, could potentially save a massive amount of disk space
    and lookup time, since the only updates needed are those for the state
    features.
    
    Cons:
    Will likely not work well for small domains, where all possible states
    can easily be cached and/or there aren't enough state features for the
    function to make use of.
    
    (gamma) - reward discount factor (between 0 and 1)
    (alpha) - learning rate (between 0 and 1)
    (epsilon) - parameter for the epsilon-greedy policy (between 0 and 1)
    (lambda) - parameter for the SARSA(lambda) learning algorithm
    """
    
    def __init__(self, alpha=0.1, gamma=1.0, epsilon=0.1, use_traces=False, *args, **kwargs):
        super(SARSALFAAgent, self).__init__(*args, **kwargs)
        #self.Q = {} # {s:{a:q-value}} default 0
        self.alpha = alpha
        self.gamma = gamma
        self._epsilon = self.epsilon = epsilon
        self.epsilon_decay_factor = 0.5
        self.episodes = 0
        
        self.use_traces = use_traces
        self.lambda_discount = kwargs.get('lambda_discount', 0.9)
        self.every_step = False
        
        #TODO:remove? only hurts performance?
        self.use_final_reward = False
        
        # A list of weights corresponding to each state parameter,
        # used as input to the linear function approximator.
        self._theta = {} # {action:[function weights for states using action]}
        
        self.reset()
    
    def __setstate__(self, d=None):
        d = d or {}
        self.__dict__.update(d)
        
    def __getstate__(self):
        return dict(
            _theta=self._theta,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=self.epsilon,
            epsilon_decay_factor=self.epsilon_decay_factor,
            episodes=self.episodes,
            use_traces=self.use_traces,
            every_step=self.every_step,
        )
        
    def normalize_state(self, state):
        """
        Converts state into a list of numbers that can be used in
        a linear function approximation.
        """
        raise NotImplementedError
    
    def reset(self):
        """
        Clears episodic variables.
        """
        
        # [(state, action, q)] of actual action selected
        self.history = []
        
        # [(state, best_action, best_q)] optimal, even if we didn't select it
#        self.history_best = []
        
        self.rewards = []
        self._epsilon = \
            self.epsilon/(1+self.episodes*self.epsilon_decay_factor)
        self.episodes += 1
    
    def get_theta(self, action, state):
        """
        Retrieves the theta vector for the given action.
        State is also required in case the vector has to be initialized.
        State should be the post-processed quantified form.
        """
        if action not in self._theta:
            self._theta[action] = [0.0]*len(state)
        theta = self._theta[action]
        assert len(theta) == len(state), \
            'State/theta mismatch. Variable state length not supported.'
        return theta
    
    def set_theta(self, action, state, theta):
        assert len(theta) == len(state), \
            'State/theta mismatch. Variable state length not supported.'
        self._theta[action] = theta
    
    def get_action(self, state, actions):
        """
        Retrieves the agent's action for the given state.
        
        state := an iterable of features, the length of which should not change
        actions := an iterable of legal actions to choice from
        """
        
        state = [1.0] + self.normalize_state(state)
        
        # Select action according to an epsilon-greedy strategy.
        if random.random() > self._epsilon:
            # Exploit our Q-values.
#            print>>sys.stderr,'exploit'
            #best = (-1e999999999999, None)
            best = ArgMaxSet()
#            print>>sys.stderr,'actions:',actions
            for _action in actions:
                theta = self.get_theta(action=_action, state=state)
                assert len(theta) == len(state)
                q = sum(
                    weight*feature
                    for weight, feature in zip(theta, state)
                )
                #best = max(best, (q, _action))
#                print>>sys.stderr, 'theta:', theta
#                print>>sys.stderr, q, _action
                best.add(score=q, obj=_action)
            #q, action = best
            q = best.score
            action = best.obj
        else:
            # Otherwise explore something random.
#            print>>sys.stderr,'explore'
            action = random.choice(actions)
            
            theta = self.get_theta(action=action, state=state)
            assert len(theta) == len(state)
            q = sum(
                weight*feature
                for weight, feature in zip(theta, state)
            )
        
        if action is None and actions:
            raise Exception('No action found!')
        
        # Record (state,action) tuple for later learning update.
        self.record_history((state, action, q))
        
        return action
        
    def reinforce(self, feedback, state=None, replace_last=False, end=None):
        """
        Processes a feedback signal (e.g. reward or punishment).
        
        http://artint.info/html/ArtInt_272.html
        
        δ = r+γQ(s',a')-Q(s,a)
        wi ← wi+ηδFi(s,a)
        
        η = eta, same as alpha
        γ = gamma
        r = reward
        """
        
        def update_step(s0, a0, q0, r1, s1, a1, q1, discount=1):
            # δ = r+γ*Qw(s',a')-Qw(s,a)
            delta = r1 + self.gamma*q1 - q0
            theta0 = self.get_theta(action=a0, state=s0)
            assert len(s0) == len(theta0)
            for i, _ in enumerate(s0):
                # wi ← wi + ηδFi(s,a)
                theta0[i] = theta0[i] + self.alpha*delta*s0[i]*discount
            self.set_theta(action=a0, state=s0, theta=theta0)
        
        if state:
            state = [1.0] + self.normalize_state(state)
        
        # Record feedback.
        if replace_last and not isinstance(self.history[-1], tuple):
            # Overwrite our last reward, since it was incorrect as our
            # opponent did something to effect it in hindsight.
            assert not isinstance(self.history[-1], tuple), \
                'Incorrect feedback replacement: %s' % (self.history[-5:],)
            self.history[-1] = feedback
            self.rewards[-1] = feedback
        else:
            if self.history:
                assert not isinstance(self.history[-1], float), \
                    'Double feedback append.'
            self.history.append(feedback)
            self.rewards.append(feedback)
            
        # Insert the null state-action pair to terminate the episode.
        if end:
            assert state is not None
            self.history.append((state, None, 0.0))
        
        # Update the function approximation weights using the SARSA algorithm.
        # http://en.wikipedia.org/wiki/SARSA
        # http://artint.info/html/ArtInt_272.html
        if len(self.history) >= 3:
            if self.use_traces:
                if end:
                    # Update weights for the episode's entire eligibility trace.
                    #TODO:this assumes a finite episode where reward is given
                    #at the end, generalize?
                    history = list(self.history)
                    final_reward = self.rewards[-1]
                    hl = len(history)
                    discount = 1
                    #print>>sys.stderr, 'history:',self.history[-10:]
                    for i in range(0, hl-2, 2):
                        # Iterate from most recent step to oldest.
                        (s0, a0, q0), r1, (s1, a1, q1) = self.history[hl-3-i:hl-i]
                        #update_step(s0, a0, q0, r1, s1, a1, q1)
                        if self.use_final_reward:
                            r1 = final_reward
                        update_step(s0, a0, q0, r1, s1, a1, q1, discount)
                        discount *= self.gamma*self.lambda_discount
                elif self.every_step:
                    history = list(self.history)[:-1]
                    final_reward = self.rewards[-1]
                    hl = len(history)
                    discount = 1
                    for i in range(0, hl-2, 2):
                        # Iterate from most recent step to oldest.
                        (s0, a0, q0), r1, (s1, a1, q1) = self.history[hl-3-i:hl-i]
                        #update_step(s0, a0, q0, r1, s1, a1, q1)
                        if self.use_final_reward:
                            r1 = final_reward
                        update_step(s0, a0, q0, r1, s1, a1, q1, discount)
                        discount *= self.gamma*self.lambda_discount
            else:
                # Update the weight for the last step.
#                print 'history:',self.history
                if end:
                    (s0, a0, q0), r1, (s1, a1, q1) = self.history[-3:]
                else:
                    (s0, a0, q0), r1, (s1, a1, q1) = self.history[-4:-1]
                update_step(s0, a0, q0, r1, s1, a1, q1)

class ANNAgent(Agent):
    """
    Uses a feed-forward artificial neural network to learn.
    
    Works by using the ANN as a function approximator of the Q-table,
    the lookup estimates the reward received in each state.
    
    So this doesn't directly associate actions to rewards.
    Instead, it associates states to rewards, and uses its transition model
    to find all next possible states, and return the action associated with the state
    with the highest Q-value.
    
    epsilon := the amount by which the agent randomly explores at first
    """
    
    def __init__(self,
        lengths,
        hiddenclass=None,
        outclass=None,
        epsilon=0.5,
        lambda_decay=0.9,
        alpha=0.01,
        *args, **kwargs):
            
        super(ANNAgent, self).__init__(*args, **kwargs)
        
        #http://pybrain.org/docs/api/structure/modules.html
        self.hiddenclass = hiddenclass or 'SigmoidLayer'
        hiddenclass = getattr(pybrain_structure, self.hiddenclass)
        
        self.outclass = outclass or 'SigmoidLayer'
        outclass = getattr(pybrain_structure, self.outclass)
        
        self.lengths = lengths
        self.network = buildNetwork(
            *lengths,
            hiddenclass=hiddenclass,
            outclass=outclass)
    
        # The exploitation/exploration rate.
        # 0=always exploit, 1=always explore
        self._epsilon = self.epsilon = epsilon
        self.epsilon_decay_factor = 0.5
        
        self.alpha = alpha
        
        # When training, this is how much the reward is diminished
        # per history step. 
        # 0=no propagation, only the last step updated
        # 1=all steps updated equally
        self.lambda_decay = lambda_decay
        
        self.episodes = 0
        self.reset()
    
    @property
    def name(self):
        return 'ANN(alpha=%s, hidden=%i, hiddenclass=%s, outclass=%s)' % (
            self.alpha,
            self.lengths[1],
            self.hiddenclass[:-5],
            self.outclass[:-5],
        )
    
    def __setstate__(self, d=None):
        d = d or {}
        self.__dict__.update(d)
        
    def __getstate__(self):
        return dict(
            alpha=self.alpha,
            lengths=self.lengths,
            network=self.network,
            epsilon=self.epsilon,
            hiddenclass=self.hiddenclass,
            outclass=self.outclass,
            epsilon_decay_factor=self.epsilon_decay_factor,
            lambda_decay=self.lambda_decay,
            episodes=self.episodes,
        )
        
    def reset(self):
        """
        Clears episodic variables.
        """
        self.history = []
        self.rewards = []
        
        # Increment our episode knowledge so we'll gradually stop exploring
        # over time and use our learned policy.
        self._epsilon = \
            self.epsilon/(1+self.episodes*self.epsilon_decay_factor)
        self.episodes += 1
    
    def normalize_state(self, state):
        """
        This should convert the state into a list of numbers that correspond
        in length to the network's input layer.
        """
        raise NotImplementedError
    
    def simulate_action(self, state, action):
        """
        Returns the expected next-state if the given action is performed
        in the given state.
        """
        raise NotImplementedError
    
    def get_action(self, state, actions):
        """
        Retrieves the agent's action for the given state.
        """
        
        # Ensure input layer matches state vector.
        assert len(state) == self.lengths[0]
        
        # Select action according to an epsilon-greedy strategy.
        if self.best or random.random() > self._epsilon:
            # Exploit our knowledge by activating network for each action
            # and using the action that corresponds to the highest Q-value.
            best = (-1e9999999, None, None)
            for action in actions:
                next_state = self.simulate_action(state, action)
                next_state = self.normalize_state(next_state)
                q_value = self.network.activate(next_state)
                best = max(best, (q_value, action, next_state))
            best_q, best_action, best_state = best
            action = best_action
        else:
            # Otherwise explore something random.
            action = random.choice(actions)
            best_state = self.simulate_action(state, action)
            best_state = self.normalize_state(best_state)
        
        # Record (state,action) tuple for later learning update.
        self.history.append(best_state)
        
        return action
    
    def reinforce(self, feedback, state=None, replace_last=False, end=None):
        """
        Processes a feedback signal (e.g. reward or punishment).
        
        Applies a feedback to every state in the episode decayed proportionally to how
        long ago the action happened.
        
        The network will be taught to associate the state right before the feedback
        with the feedback most strongly, whereas a state at the beginning of the episode
        will have a very weak association.
        """
        
        if not end:
            return
        
        self.rewards.append(feedback)
        
        # Build training set.
        #ds = SupervisedDataSet(inp=9, target=1)
        ds = SupervisedDataSet(inp=self.lengths[0], target=self.lengths[-1])
        
        r0 = feedback
        for normalized_state in reversed(self.history):
#            print 'sample:',normalized_state,r0
            ds.addSample(normalized_state, (r0,))
            r0 *= self.lambda_decay
        
        trainer = BackpropTrainer(
            self.network, ds, learningrate=self.alpha)
        trainer.train()

class PIDNN(ANNAgent):
    """
    A PID controller using a neural network to learn components.
    
    http://molefrog.com/pidnn-talk
    """
    
    def __init__(self, *args, **kwargs):
        lengths = [2, 3, 1]
        super(PIDNN, self).__init__(lengths=lengths, *args, **kwargs)
        self._target = 0
        self._feedback = 0

    def get_action(self, *args, **kwargs):
        raise NotImplementedError
        
    @property
    def name(self):
        return 'PIDNN(alpha=%s, hidden=%i, hiddenclass=%s, outclass=%s)' % (
            self.alpha,
            self.lengths[1],
            self.hiddenclass[:-5],
            self.outclass[:-5],
        )
        
    @property
    def target(self):
        return self._target
         
    @target.setter
    def target(self, v):
        self._target = v

    @property
    def feedback(self):
        return self._feedback
        
    @feedback.setter
    def feedback(self, v):
        self._feedback = v

    def get_output(self):
        
        inputs = [
            self._target, # desired system output
            self._feedback, # last system output
        ]
        
        output = self.network.activate(inputs)
        
        ds = SupervisedDataSet(inp=self.lengths[0], target=self.lengths[-1])
        ds.addSample(inputs, self._target)
        
        trainer = BackpropTrainer(self.network, ds, learningrate=self.alpha)
        trainer.train()
        
        #self._feedback = output
        
        return output
