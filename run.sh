#!/bin/bash
./run_experiments.sh --dataset "$1" --local none  --crossover two_point  --crossitems 10 --crossinterval 5 --runs "${2:-10}" --print-every 1

