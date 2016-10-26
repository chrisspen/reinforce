Reinforce - A simple Python reinforcement learning library.
===========================================================


Overview
-----------


Installation
---------------

Install the package using pip:

    pip install reinforce

Usage
--------

Development
-----------

To run all [tests](http://tox.readthedocs.org/en/latest/):

    export TESTNAME=; tox
    
To run tests for a specific environment (e.g. Python 2.7 with Django 1.4):
    
    export TESTNAME=; tox -e py27

To run a specific test:
    
    export TESTNAME=.test_xo_lfa; tox -e py27
