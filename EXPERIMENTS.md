# FcCon Bash experiments

Build: `qmake FcCon.pro && make -j$(nproc)`

Single pair:
```bash
./run_experiments.sh --exe ./FcCon --train data.train --test data.test --features 2 --crossover targeted --mutation standard --runs 30 --out results/targeted.csv
```

All 25 pairs, 30 seeds each:
```bash
./run_experiments.sh --exe ./FcCon --train data.train --test data.test --features 2 --crossover all --mutation all --runs 30 --seed 1 --generations 200 --chromosomes 500 --length 100 --trials 20 --out results/full.csv
```

One CSV row per execution; complete stdout/stderr in separate logs. All operator pairs reuse the same seeds. Requires Bash and coreutils (realpath). The targeted operators are fitness-guided approximations, not identical to QGenClass class-aware operators.

## No-operator ablation baselines

The `none` choice is available for both `--crossover` and `--mutation`. `--crossover none` skips the population-level reproduction crossover operator; `--mutation none` skips the population-level mutation operator. The original `standard` operators remain the defaults.

```bash
./run_experiments.sh --dataset iris --crossover standard --mutation standard --runs 30
./run_experiments.sh --dataset iris --crossover none --mutation standard --runs 30
./run_experiments.sh --dataset iris --crossover standard --mutation none --runs 30
./run_experiments.sh --dataset iris --crossover none --mutation none --runs 30
```

`--crossover all --mutation all` now runs 6 × 6 = 36 combinations (1,080 runs for `--runs 30`). Note: FcCon has separate local-search/cross-item routines, including periodic cross-item operations, which are **not** disabled by `--crossover none`. Thus `none/none` is a no-global-crossover/no-global-mutation ablation, not necessarily a fully frozen population.
