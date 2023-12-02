#!/bin/bash

autopep8 --in-place --aggressive --aggressive --max-line-length 127 src/*.py *.py
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude .git,__pycache__,dataset,bertweet,xlm-roberta
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --exclude .git,__pycache__,dataset,bertweet,xlm-roberta