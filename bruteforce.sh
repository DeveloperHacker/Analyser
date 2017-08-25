#!/usr/bin/env bash
PYTHONPATH="src/"
while true; do
    python src/analyser_main.py --random=true --train=true
done
