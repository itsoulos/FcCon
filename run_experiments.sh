#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'HELP'
Usage: ./run_experiments.sh --dataset NAME [options]
       ./run_experiments.sh --train FILE --test FILE [options]
  --exe PATH              Executable (default ./FcCon)
  --data-dir DIR          Dataset folder (default ~/Desktop/ERGASIES/FeatureConstruction2/datasets/tenfolding)
  --dataset NAME          Dataset basename (resolves NAME.train and NAME.test)
  --train FILE            Training file (overrides --dataset)
  --test FILE             Test file (overrides --dataset)
  --features N            Constructed features (default 2)
  --runs N                Repetitions per pair (default 10)
  --generations N         Generations (default 500)
  --print-every N         Print progress every N generations (default 10)
  --chromosomes N         Population size (default 500)
  --length N              Chromosome length (default 100)
  --model NAME            Model (default rbf)
  --fitness neural|hybrid|neural_rules Fitness mode (default neural)
  --rule-weight FLOAT      Hybrid rule weight (default 0.25)
  --rules-tolerance FLOAT  Neural error bucket width (default 0.01)
  --local NAME            Local search (none, adam, discrete_adam, etc.; default none)
  --trials N              Operator trials (default 20)
  --crossitems N          Random chromosomes for periodic crossItem (default 0)
  --crossinterval N       Apply crossItem every N generations (default 0)
  --crossover NAME|all    none,standard,two_point,uniform,targeted,targetedWorst,constants
  --mutation NAME|all     none,standard,creep,adaptive,targeted,targetedWorst
  --seed N                Initial seed (default 1)
  --out FILE              CSV results (default results/fccon_results.csv)
  --summarize-only        Recalculate summary from existing CSV/logs (no FcCon runs)
  --help                  Show help
  Summary: results/fccon_results_summary.csv (mean and sample STD of CLASS error %)
HELP
}
exe='./FcCon'; data_dir="$HOME/Desktop/ERGASIES/FeatureConstruction2/datasets/tenfolding"; dataset=''; train=''; test=''; features=2; runs=10; generations=500
chromosomes=500; length=100; print_every=10; model='rbf'; local='none'; fitness='neural'; rule_weight=0.25; rules_tolerance=0.01; trials=20
crossitems=0; crossinterval=0; crossover='standard'; mutation='standard'; seed=1; out='results/fccon_results.csv'; summarize_only=0
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --summarize-only) summarize_only=1; shift ;;
    --exe|--data-dir|--dataset|--train|--test|--features|--runs|--generations|--print-every|--chromosomes|--length|--model|--local|--fitness|--rule-weight|--rules-tolerance|--trials|--crossitems|--crossinterval|--crossover|--mutation|--seed|--out)
      if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
      key=${1#--}; key=${key//-/_}; printf -v "$key" '%s' "$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
for key in features runs generations print_every chromosomes length trials; do
  value=${!key}
  if [[ ! $value =~ ^[0-9]+$ ]] || ((10#$value < 1)); then echo "Invalid $key: $value" >&2; exit 2; fi
done
for key in crossitems crossinterval; do
  value=${!key}
  if [[ ! $value =~ ^[0-9]+$ ]]; then echo "Invalid $key: $value" >&2; exit 2; fi
done
if [[ ! $seed =~ ^[0-9]+$ ]]; then echo "Invalid seed: $seed" >&2; exit 2; fi
# Resolve dataset names relative to the default (or overridden) directory.
# Explicit --train/--test paths take precedence over --dataset.
if [[ -n $dataset ]]; then
  [[ -n $train ]] || train="$dataset.train"
  [[ -n $test ]] || test="$dataset.test"
fi
for key in train test; do
  value=${!key}
  if [[ -n $value && ! -f $value && -f $data_dir/$value ]]; then
    printf -v "$key" '%s' "$data_dir/$value"
  fi
done
if (( ! summarize_only )) && [[ -z $train || -z $test || ! -f $train || ! -f $test ]]; then
  echo "Training/test files not found. Dataset directory: $data_dir" >&2
  echo 'Use --dataset NAME, or --train FILE --test FILE, optionally with --data-dir DIR.' >&2
  exit 2
fi
if (( ! summarize_only )) && [[ ! -f $exe || ! -x $exe ]]; then echo "Executable not found/not executable: $exe" >&2; exit 2; fi
allowed=(none standard two_point uniform targeted targetedWorst constants)
mut_allowed=(none standard creep adaptive targeted targetedWorst)
contains() { local needle=$1 x; shift; for x in "$@"; do [[ $needle == "$x" ]] && return 0; done; return 1; }
if [[ $crossover != all ]] && ! contains "$crossover" "${allowed[@]}"; then echo "Invalid crossover: $crossover" >&2; exit 2; fi
if [[ $mutation != all ]] && ! contains "$mutation" "${mut_allowed[@]}"; then echo "Invalid mutation: $mutation" >&2; exit 2; fi
if [[ $crossover == all ]]; then cross_list=("${allowed[@]}"); else cross_list=("$crossover"); fi
if [[ $mutation == all ]]; then mut_list=("${mut_allowed[@]}"); else mut_list=("$mutation"); fi
# Keep relative output names under results/; preserve explicit absolute paths.
if [[ $out != /* && $out != results/* ]]; then
  out="results/$out"
fi
mkdir -p -- "$(dirname -- "$out")" || exit 2
base=${out%.*}; [[ $base == "$out" ]] && base=$out
summary="${base}_summary.csv"
# Accept CLASS on the heading line or following lines, including ANSI/CR output.
extract_errors() {
  awk '''
    { gsub(/\033\[[0-9;]*[[:alpha:]]/, "", $0); gsub(/\r/, "", $0) }
    /NEURAL[[:space:]]+AVERAGES/ { section="mlp" }
    /RBF[[:space:]]+AVERAGES/ { section="rbf" }
    section != "" && /CLASS[[:space:]]*:/ {
      line=$0
      sub(/^.*CLASS[[:space:]]*:[[:space:]]*/, "", line)
      if (match(line, /^[-+]?[0-9]+([.][0-9]+)?/)) {
        value=substr(line,RSTART,RLENGTH)
        if (section=="mlp") mlp=value
        else rbf=value
      }
      section=""
    }
    END { printf "%s,%s\n", mlp, rbf }
  ''' "$1"
}
record_log() {
  local c=$1 m=$2 current_seed=$3 status=$4 log=$5 mlp="" rbf=""
  if [[ $status == 0 && -f $log ]]; then
    IFS=, read -r mlp rbf < <(extract_errors "$log")
    if [[ ! $mlp =~ ^[0-9]+([.][0-9]+)?$ || ! $rbf =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      echo "WARNING: Could not parse both CLASS errors from $log" >&2
      mlp=""; rbf=""
    fi
  fi
  printf "%s,%s,%s,%s,%s,%s,%s\n" "$c" "$m" "$current_seed" "$status" "$log" "$mlp" "$rbf" >> "$out"
}
if (( summarize_only )); then
  [[ -f $out ]] || { echo "CSV not found: $out" >&2; exit 2; }
  old_csv=$(mktemp)
  cp -- "$out" "$old_csv"
  printf "crossover,mutation,seed,returncode,log,mlp_class_error_pct,rbf_class_error_pct\n" > "$out"
  while IFS=, read -r c m current_seed status log _; do
    [[ $c == crossover || -z $c ]] && continue
    record_log "$c" "$m" "$current_seed" "$status" "$log"
  done < "$old_csv"
  rm -f -- "$old_csv"
  failed=0
else
  printf "crossover,mutation,seed,returncode,log,mlp_class_error_pct,rbf_class_error_pct\n" > "$out"
  exe=$(realpath -- "$exe"); train=$(realpath -- "$train"); test=$(realpath -- "$test")
  failed=0
  for c in "${cross_list[@]}"; do
    for m in "${mut_list[@]}"; do
      for ((run=0;run<runs;run++)); do
        current_seed=$((10#$seed + run))
        log="${base}_${c}_${m}_seed${current_seed}.log"
        echo "$c / $m / seed $current_seed"
        "$exe" "--fc_trainfile=$train" "--fc_testfile=$test" \
          "--fc_dimension=$features" '--fc_iters=1' "--fc_seed=$current_seed" \
          "--fc_generations=$generations" "--fc_print_every=$print_every" "--fc_chromosomes=$chromosomes" \
          "--fc_length=$length" "--fc_model=$model" "--fc_local=$local" "--fc_fitness=$fitness" "--fc_rule_weight=$rule_weight" "--fc_rules_tolerance=$rules_tolerance" \
          "--fc_crossover=$c" "--fc_mutation=$m" "--fc_operator_trials=$trials" \
          "--fc_crossitems=$crossitems" "--fc_crossinterval=$crossinterval" \
          2>&1 | tee "$log"
        status=${PIPESTATUS[0]}
        record_log "$c" "$m" "$current_seed" "$status" "$log"
        if ((status != 0)); then echo "FAILED ($status): $log" >&2; failed=$((failed+1)); fi
      done
    done
  done
fi
# Group by crossover/mutation. Exclude failed runs and incomplete metrics.
awk -F, -v OFS=, '
  NR == 1 { next }
  $4 == 0 && $6 ~ /^[0-9]+([.][0-9]+)?$/ && $7 ~ /^[0-9]+([.][0-9]+)?$/ {
    key=$1 SUBSEP $2
    n[key]++
    sm[key]+=$6; sr[key]+=$7
    qm[key]+=$6*$6; qr[key]+=$7*$7
    if (!(key in seen)) { seen[key]=1; order[++total]=key }
  }
  function std(sum,sumsq,count, variance) {
    if (count < 2) return 0
    variance=(sumsq-sum*sum/count)/(count-1)
    return sqrt(variance > 0 ? variance : 0)
  }
  BEGIN { print "crossover","mutation","valid_runs","mlp_mean_pct","mlp_std_pct","rbf_mean_pct","rbf_std_pct" }
  END {
    for (i=1;i<=total;i++) {
      key=order[i]; split(key,parts,SUBSEP)
      printf "%s,%s,%d,%.6f,%.6f,%.6f,%.6f\n",parts[1],parts[2],n[key],sm[key]/n[key],std(sm[key],qm[key],n[key]),sr[key]/n[key],std(sr[key],qr[key],n[key])
    }
  }
' "$out" > "$summary"
printf '\nSUMMARY (classification error %%; sample STD):\n'
if command -v column >/dev/null 2>&1; then column -s, -t "$summary"; else cat "$summary"; fi
printf '\nFinished. CSV: %s; summary: %s; failed runs: %d\n' "$out" "$summary" "$failed"
((failed == 0))
