BASEPATH=~/Desktop/ERGASIES/FeatureConstruction2/
PROGRAM=./FcCon
DATAPATH=$BASEPATH/datasets/tenfolding/
#DATAPATH=~/Desktop/ERGASIES/MLFOREST/MONO_DOUBLE/DOUBLE/TENFOLDING/DATA/
## Number of iterations
ITERS=1
## Number of allowed generations
GENS=500
## Number of chromosomes
COUNT=500
## Length of each chromosome
LENGTH=200
## Dataset name
DATAFILE=$1
## Number of constructed features
DIMENSION=$2
## Number of weights used in the construction process
WEIGHTS=3
## The name of  input train file
TRAINFILE=$DATAPATH/$DATAFILE.train
## The name of  input test file
TESTFILE=$DATAPATH/$DATAFILE.test
## The model used for feature construction
MODEL=rbf ##Values: rbf,neural,airbf
## The local optimization procedure
LOCAL=mutate ##Values: none,crossover,mutate,de,siman,gd,adam
## Enable or disable the balanced class fitness
BALANCECLASS=yes
## Enable or disable the usage of SMOTE
ENABLESMOTE=no
## Enable or disable the usage of Geometric Mean
ENABLEMEAN=no
## Enable or disable the usage of class fitness instead of mse fitness. 
## Useful for classification problems.
ENABLECLASSFITNESS=no
## Enable or disable the normalization with min - max of the dataset.
ENABLENORM=yes
 $PROGRAM --fc_iters=$ITERS --fc_generations=$GENS --fc_chromosomes=$COUNT --fc_length=$LENGTH --fc_dimension=$DIMENSION --fc_weights=$WEIGHTS --fc_trainfile=$TRAINFILE --fc_testfile=$TESTFILE --fc_local=$LOCAL --fc_balanceclass=$BALANCECLASS --fc_enablesmote=$ENABLESMOTE --fc_enablemean=$ENABLEMEAN --fc_model=$MODEL --fc_enableclassfitness=$ENABLECLASSFITNESS --fc_enablenorm=$ENABLENORM
