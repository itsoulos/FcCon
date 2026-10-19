#!/bin/bash
./run_experiments.sh --dataset "$1" --local mutate  --crossover standard  --crossitems 5 --crossinterval 1 --runs "${2:-10}" --print-every 1
