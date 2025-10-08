#!/bin/sh
set -eu

PROJECT="ntu-music-singer-cnn"
ENTITY="seanfu920506-national-taiwan-university"
COUNT=20

echo "Creating and launching sweep via Python helper..."
python3 run_sweep.py --yaml sweep.yaml --project "$PROJECT" --entity "$ENTITY" --count "$COUNT"