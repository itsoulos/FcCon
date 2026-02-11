BASEPATH=~/Desktop/ERGASIES/FcCon/
PROGRAM=$BASEPATH/FcCon
DATAPATH=$BASEPATH/datasets/tenfolding/
## Number of iterations
ITERS=10
## Number of allowed generations
GENS=200
## Number of chromosomes
COUNT=500
## Length of each chromosome
LENGTH=100
## Dataset name
DATAFILE=$1
## Number of constructed features
DIMENSION=$2
## Number of weights used in the construction process
WEIGHTS=10
## The name of  input train file
TRAINFILE=$DATAPATH/$DATAFILE.train
## The name of  input test file
TESTFILE=$DATAPATH/$DATAFILE.test
## The model used for feature construction
MODEL=rbf ##Values: rbf,neural
## The local optimization procedure
LOCAL=none ##Values: none,crossover,mutate,de
## Enable or disable the balanced class fitness
BALANCECLASS=no
## Enable or disable the usage of SMOTE
ENABLESMOTE=yes
$PROGRAM --fc_iters=$ITERS --fc_generations=$GENS --fc_chromosomes=$COUNT --fc_length=$LENGTH --fc_dimension=$DIMENSION --fc_weights=$WEIGHTS --fc_trainfile=$TRAINFILE --fc_testfile=$TESTFILE --fc_local=$LOCAL --fc_balanceclass=$BALANCECLASS --fc_enablesmote=$ENABLESMOTE
