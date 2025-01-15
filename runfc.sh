BASEPATH=~/Desktop/ERGASIES/FcCon/
PROGRAM=$BASEPATH/FcCon
DATAPATH=$BASEPATH/datasets/tenfolding/
ITERS=10
GENS=200
COUNT=500
LENGTH=100
DATAFILE=$1
DIMENSION=$2
WEIGHTS=10
TRAINFILE=$DATAPATH/$DATAFILE.train
TESTFILE=$DATAPATH/$DATAFILE.test
MODEL=rbf ##Values: rbf,neural
LOCAL=de ##Values: none,crossover,mutate,de
$PROGRAM --fc_iters=$ITERS --fc_generations=$GENS --fc_chromosomes=$COUNT --fc_length=$LENGTH --fc_dimension=$DIMENSION --fc_weights=$WEIGHTS --fc_trainfile=$TRAINFILE --fc_testfile=$TESTFILE --fc_local=$LOCAL
