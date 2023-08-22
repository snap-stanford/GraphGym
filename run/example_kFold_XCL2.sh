#!/usr/bin/env bash

# Test for running a single experiment. --repeat means run how many different random seeds.
python main.py --cfg configs/example_kFold_XCL2.yaml --repeat 1
