#!/usr/bin/env bash
PYTHONPATH="src/"
python src/boot/generate_embeddings.py
python src/boot/main.py