#!/usr/bin/env bash
PYTHONPATH="src/"
python src/word2vec_main.py
python src/analyser_main.py --prepare=true --train=true --cross=true  --test=true
