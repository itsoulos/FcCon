# FcCon: Final Experimental Guide (Crossover, Mutation, and Bash Runner)

## Overview

FcCon performs **feature construction using Grammatical Evolution (GE)**. This experimental extension adds selectable crossover and mutation operators, fitness-guided variants inspired by the QGenClass project, a Bash experiment runner, and configurable generation-by-generation progress output.

The original `standard` operators remain available as a baseline. The extension does not claim that its fitness-guided operators are exact copies of QGenClass's class-aware semantic operators: FcCon uses its feature-construction fitness instead of QGenClass's per-class error information.

## Build and requirements

- Linux with Bash and GNU coreutils (`realpath`)
- A C++17 compiler, Qt development packages, `qmake`, and `make`, together with any other dependencies required by the FcCon project

From the FcCon project directory:

```bash
qmake FcCon.pro
make -j"$(nproc)"
chmod +x run_experiments.sh
```

The runner expects an executable at `./FcCon` by default. Override this with `--exe PATH` if your build produces a different path.

> **Validation status:** The previous package was syntax-checked as a Bash script, but a complete C++ build and end-to-end experiment were not verified in the provided environment. Check the build and a single short run before launching large experiment grids.

## Dataset layout

By default, the Bash script looks for datasets in:

```text
~/Desktop/ERGASIES/FeatureConstruction2/datasets/tenfolding/
```

For `--dataset iris`, the script resolves:

```text
iris.train
iris.test
```

Both files must exist. You can override the directory with `--data-dir DIR`, or specify explicit paths with `--train FILE --test FILE`. The script runs **one named dataset pair at a time**; `--dataset all` is not implemented in this version.

## Crossover operators

| Name | Description |
|---|---|
| `none` | Disable the main crossover operator (other local-search mechanisms may still act). |
| `standard` | Original one-point crossover baseline. |
| `two_point` | Recombination using two selected crossover positions. |
| `uniform` | Gene-wise recombination controlled by random choices. |
| `targeted` | Fitness-guided block transfer from a donor selected among better-ranked chromosomes. |
| `targetedWorst` | Fitness-guided block transfer using a worse-ranked chromosome as donor. |

The targeted variants evaluate trial offspring using FcCon's fitness function and retain accepted improvements. `targetedWorst` refers to the **donor chromosome**, not the worst-performing classification class.

## Mutation operators

| Name | Description |
|---|---|
| `none` | Disable the main mutation operator (other local-search mechanisms may still act). |
| `standard` | Original random gene replacement baseline. |
| `creep` | Small incremental gene changes (±1, wrapped within the rule range). |
| `adaptive` | Incremental changes whose maximum step depends on the generation index. |
| `targeted` | Fitness-guided trials of single-gene replacements. |
| `targetedWorst` | Fitness-guided trials of larger mutated gene blocks. |

Fitness-guided operators perform up to `--trials N` trial modifications when invoked. Their extra fitness evaluations can increase running time; compare both predictive performance and computational cost.

## Bash runner options

Run `./run_experiments.sh --help` to display the command-line reference.

| Option | Default | Meaning |
|---|---|---|
| `--exe PATH` | `./FcCon` | FcCon executable |
| `--data-dir DIR` | `~/Desktop/ERGASIES/FeatureConstruction2/datasets/tenfolding` | Dataset directory |
| `--dataset NAME` | none | Resolve `NAME.train` and `NAME.test` |
| `--train FILE` | none | Explicit training file |
| `--test FILE` | none | Explicit test file |
| `--features N` | `1` | Number of constructed features |
| `--runs N` | `30` | Independent runs per operator combination |
| `--generations N` | `200` | GE generations |
| `--print-every N` | `10` | Progress reporting interval in generations |
| `--chromosomes N` | `500` | Population size |
| `--length N` | `100` | Chromosome length |
| `--model NAME` | `rbf` | Model used in feature construction |
| `--local NAME` | `none` | Local-search method |
| `--trials N` | `20` | Fitness-guided operator trial budget |
| `--crossover NAME\|all` | `standard` | Crossover choice or all six choices |
| `--mutation NAME\|all` | `standard` | Mutation choice or all six choices |
| `--seed N` | `1` | Starting seed |
| `--out FILE` | `fccon_results.csv` | Run manifest CSV |

Supported local-search names in the current C++ configuration are `none`, `crossover`, `mutate`, `de`, `siman`, `gd`, and `adam`. Model availability depends on the underlying FcCon build.

## Interpreting the `none` controls

`--crossover none` disables the main crossover operator; `--mutation none` disables the main mutation operator. They can be used independently or together. These switches do **not** necessarily disable separate local-search operations or the existing periodic `crossItem` mechanism. Thus, `none/none` should not be interpreted as a completely static population. For a fair baseline, use `standard/standard` as the original-algorithm reference and `none/standard` or `standard/none` for ablation studies.

## Usage examples

### 1. Standard baseline

```bash
./run_experiments.sh \
  --dataset iris \
  --features 2 \
  --crossover standard \
  --mutation standard \
  --runs 30 \
  --print-every 10 \
  --out results/baseline.csv
```

### 2. Crossover ablation (no main crossover)

```bash
./run_experiments.sh \
  --dataset iris \
  --features 2 \
  --crossover none \
  --mutation standard \
  --runs 30 \
  --out results/no_crossover.csv
```

### 3. Compare all crossover operators with standard mutation

```bash
./run_experiments.sh \
  --dataset iris \
  --features 2 \
  --crossover all \
  --mutation standard \
  --runs 30 \
  --print-every 20 \
  --out results/crossover_comparison.csv
```

This schedules **6 × 30 = 180 runs**.

### 4. Compare all mutation operators with standard crossover

```bash
./run_experiments.sh \
  --dataset iris \
  --features 2 \
  --crossover standard \
  --mutation all \
  --runs 30 \
  --out results/mutation_comparison.csv
```

### 5. Full factorial experiment

```bash
./run_experiments.sh \
  --dataset iris \
  --features 2 \
  --crossover all \
  --mutation all \
  --runs 30 \
  --seed 1 \
  --generations 200 \
  --chromosomes 500 \
  --length 100 \
  --trials 20 \
  --print-every 10 \
  --out results/full.csv
```

This schedules **6 crossover × 6 mutation × 30 repetitions = 1,080 runs** for one dataset pair.

### 6. Explicit train/test files

```bash
./run_experiments.sh \
  --train /path/to/custom.train \
  --test /path/to/custom.test \
  --features 1 \
  --crossover targeted \
  --mutation targeted \
  --runs 10
```

## Live progress and logging

The C++ application accepts `--fc_print_every=N`; the Bash script exposes it as `--print-every N`. Progress information is emitted during evolution at the configured generation interval, as well as the initial/final progress points implemented in the C++ code. The runner uses `tee` so standard output and standard error are displayed live and also saved to individual `.log` files.

The exact progress fields and numerical values depend on the current FcCon output. Do not assume the values below are literal output from a completed experiment:

```text
RUN: 1 GENERATION=10 FITNESS=...
RUN: 1 GENERATION=20 FITNESS=...
```

The runner creates a CSV **run manifest** with columns:

```csv
crossover,mutation,seed,returncode,log
```

Each row identifies the operator pair, seed, process exit status, and path to its log. **The CSV does not currently contain parsed fitness, classification accuracy, means, or standard deviations.** Those metrics must be extracted from the logs separately.

The same seed sequence is reused across operator combinations, helping controlled comparisons. A zero return code indicates process completion, not proof of statistical correctness.

## Experimental methodology

For reproducible comparisons:

1. Use the same train/test split, feature count, population size, chromosome length, generation limit, and local-search settings for all operators.
2. Keep the same sequence of seeds for each operator combination.
3. Compare predictive metrics **and** wall-clock time or fitness evaluation counts, particularly for targeted variants.
4. Report mean, standard deviation, and ideally paired statistical comparisons across repeated runs.
5. Keep the standard crossover/mutation combination as the reference baseline; treat `none` variants as ablation controls.
6. Explicitly distinguish FcCon's fitness-guided targeted variants from QGenClass's class-aware semantic operators in publications.

## Implementation notes and limitations

- The Bash runner invokes the executable once per operator pair and seed, passing the GE parameters as `--fc_*` arguments.
- The default `--runs 30` produces seeds `1` through `30` when `--seed 1` is used.
- All experiment combinations run **sequentially**, not in parallel.
- The current runner requires both train and test files to exist before starting.
- The current CSV stores execution metadata rather than parsed model-performance summaries.
- The extension should be tested on a small dataset before large runs; a full C++ build and end-to-end correctness test were not previously confirmed.

## Quick command reference

```bash
./run_experiments.sh --help
```

Project source: https://github.com/itsoulos/FcCon
