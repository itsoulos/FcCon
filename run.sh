#!/bin/bash
./run_experiments.sh --dataset "$1" --local mutate  --crossover two_point  --crossitems 10 --crossinterval 10 --runs "${2:-10}" --print-every 1  --fitness neural_rules   --rules-tolerance 0.05

