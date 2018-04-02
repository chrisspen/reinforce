Reinforce Toolkit - A simple Python reinforcement learning library.
===========================================================

[![](https://img.shields.io/pypi/v/reinforce-toolkit.svg)](https://pypi.python.org/pypi/reinforce-toolkit) [![Build Status](https://img.shields.io/travis/chrisspen/reinforce.svg?branch=master)](https://travis-ci.org/chrisspen/reinforce) [![](https://pyup.io/repos/github/chrisspen/reinforce/shield.svg)](https://pyup.io/repos/github/chrisspen/reinforce)

Overview
-----------

A collection of reinforcement learning algorithms.

Installation
---------------

Install the package using pip:

    pip install reinforce-toolkit

Usage
--------

Development
-----------

To run all [tests](http://tox.readthedocs.org/en/latest/):

    export TESTNAME=; tox
    
To run tests for a specific environment:
    
    export TESTNAME=; tox -e py27

To run a specific test:
    
    export TESTNAME=.test_xo_lfa; tox -e py27
