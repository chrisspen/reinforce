#!/bin/bash
set -e
./pep8.sh
[ -d .tox ] && rm -Rf .tox || true
tox
