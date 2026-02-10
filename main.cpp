# include <stdio.h>
# include <CORE/parameterlist.h>
# include <QDebug>
# include <GE/population.h>
# include <math.h>
# include <GE/nnprogram.h>
# include <QCoreApplication>
# include <QFile>

NNprogram *p=NULL;
Population *pop=NULL;

int random_seed=1;
int fc_iters=10;
int fc_generations=200;
int fc_chromosomes=500;
int fc_length=100;
int fc_dimension=1;
int fc_weights=10;
QString fc_model="rbf";//available values: rbf,neural;

ParameterList mainParamList;
ParameterList neuralParamList;
QString selectedTrainFile = "";
QString selectedTestFile = "";
bool    fc_balanceclass=false;

void init();
void run();
void done();

void makeMainParams()
{
    mainParamList.addParam(Parameter("help","","Print help screnn and terminate"));
    mainParamList.addParam(Parameter("fc_seed",1,1,100,"Random Seed"));
    mainParamList.addParam(Parameter("fc_iters",10,1,100,"GE iterations"));
    mainParamList.addParam(Parameter("fc_generations",200,10,1000,"Number of allowed generations"));
    mainParamList.addParam(Parameter("fc_chromosomes",500,100,1000,"Number of chromosomes"));
    mainParamList.addParam(Parameter("fc_length",100,10,1000,"Chromosome length"));
    mainParamList.addParam(Parameter("fc_dimension",1,1,10,"Number of constructed features"));
    mainParamList.addParam(Parameter("fc_weights",10,1,20,"Weights for the model used in feature construction"));
    mainParamList.addParam(Parameter("fc_trainfile","","Used train file"));
    mainParamList.addParam(Parameter("fc_testfile","","Used test file"));
    QStringList modelList;
    modelList<<"rbf"<<"neural";
    mainParamList.addParam(Parameter("fc_model",modelList[0],modelList,"Used model for feature construction"));
    QStringList localList;
    localList<<"none"<<"crossover"<<"mutate"<<"de";
    mainParamList.addParam(Parameter("fc_local",localList[0],localList,"Used local search method"));
    QStringList yesno;
    yesno<<"no"<<"yes";
    mainParamList.addParam(Parameter("fc_balanceclass",
                                     yesno[0],yesno,
                                     "Use balanced class fitness "));
}

int    neural_bfgs_iters= 2001;
double neural_left_margin=-10.0;
double neural_right_margin=10.0;
QString neural_train_method="bfgs";//available values: bfgs,genetic,lbfgs

void    makeNeuralParams()
{
    neuralParamList.addParam(Parameter("neural_bfgs_iters",2001,10,10000,"Number of bfgs iters"));
    neuralParamList.addParam(Parameter("neural_left_margin",-10.0,-100.0,0.0,"Left bound for neural"));
    neuralParamList.addParam(Parameter("neural_right_margin",10.0,0.0,100.0,"Neural right bound"));
    QStringList method;
    method<<"bfgs"<<"genetic"<<"lbfgs";
    neuralParamList.addParam(Parameter("neural_train_method",method[0],method,"Training method for neural network"));
}

void makeParams()
{
    makeMainParams();
    makeNeuralParams();
}


void printOption(Parameter param)
{
    qDebug().noquote()<<"Parameter name:           "<<param.getName();
    qDebug().noquote()<<"\tParameter default value:"<<param.getValue();
    qDebug().noquote()<<"\tParameter purpose:      "<<param.getHelp();
}

void shouldTerminate()
{
    done();
    qApp->exit(0);
    exit(0);
}

void error(QString message)
{
    printf("Fatal error %s \n",message.toStdString().c_str());
    shouldTerminate();
}

void    printHelp()
{
    qDebug().noquote()<<"MAIN PARAMETERS\n=================================================";
    for(int i=0;i<mainParamList.countParameters();i++)
       printOption(mainParamList.getParam(i));
    qDebug().noquote()<<"NEURAL PARAMETERS\n=================================================";
    for(int i=0;i<neuralParamList.countParameters();i++)
       printOption(neuralParamList.getParam(i));
    shouldTerminate();
}



void parseCmdLine(QStringList args)
{
    QString lastParam="";
    for(int i=1;i<args.size();i++)
    {
        if(args[i]=="--help") printHelp();
        QStringList lst = args[i].split("=");
        if(lst.size()<=1)
            error(QString("Fatal error %1 not an option").arg(args[i]));
        QString name = lst[0];
        QString value = lst[1];
        if(name.startsWith("--"))
            name = name.mid(2);
        if(value=="")
        {
            error(QString("Param %1 is empty.").arg(value));
        }
        bool foundParameter = false;
        //check in mainParams
        if(mainParamList.contains(name))
        {
            mainParamList.setParam(name,value);
            foundParameter=true;
        }
        if(foundParameter) continue;
        //check in neural
        if(neuralParamList.contains(name))
        {
            neuralParamList.setParam(name,value);
            foundParameter=true;
        }
        if(!foundParameter)
            error(QString("Parameter %1 not found.").arg(name));
    }
}



void    loadDataFiles()
{
    //load train and test files
    selectedTrainFile = mainParamList.getParam("fc_trainfile").getValue();
    selectedTestFile =  mainParamList.getParam("fc_testfile").getValue();

    if(selectedTrainFile.isEmpty() || selectedTestFile.isEmpty())
        error("The user should provide train and test file");

    if(!QFile::exists(selectedTrainFile))
        error(QString("Trainfile %1 does not exist").arg(selectedTrainFile));

    if(!QFile::exists(selectedTestFile))
        error(QString("Trainfile %1 does not exist").arg(selectedTestFile));

}

void    init()
{
    makeParams();
}

void green()
{
    printf("\033[1;32m");

}
void red () {
    printf("\033[1;31m");
  }


  void reset () {
    printf("\033[0m");
  }

void runNeural(int iter,double &testError,double &classError,
                 double &avg_precision,
                 double &avg_recall,
                 double &avg_fscore)
{
    char train_file[1024];
    char test_file[1024];
    int pattern_dimension = mainParamList.getParam("fc_dimension").getValue().toInt();

    strcpy(train_file,selectedTrainFile.toStdString().c_str());
    strcpy(test_file,selectedTestFile.toStdString().c_str());
    int total_runs=30;
    int w=10;
    srand(iter);
    srand48(iter);
    Neural *neural = new Neural(p->getMapper());
    neural->readPatterns(train_file);
    neural->setNumOfWeights(w);
    neural->setPatternDimension(pattern_dimension);
    classError = 0.0;
    testError = 0.0;
    avg_precision = 0.0;
    avg_recall = 0.0;
    avg_fscore = 0.0;
    for(int i=1;i<=total_runs;i++)
    {
    double d=neural->train2();
    testError+=neural->testError(test_file);
    double precision,recall,fscore;
    classError+=neural->classTestError(test_file,precision,recall);
    neural->getPrecisionRecall(test_file,precision,recall,fscore);
    avg_precision+=precision;
    avg_recall+=recall;
    avg_fscore+=fscore;
    }
    testError/=total_runs;
    classError/=total_runs;
    avg_precision/=total_runs;
    avg_recall/=total_runs;
    avg_fscore/=total_runs;
    //make red report
    red();
    printf("NEURAL. TEST ERROR: %10.5lg CLASS ERROR: %10.5lg%%\n",testError,classError*100.0);
    printf("NEURAL. PRECISION %10.5lg RECALL %10.5lg FSCORE %10.5lg\n",
           avg_precision,avg_recall,avg_fscore);
    reset();
    delete neural;
}

void runRbf(int iter,
            double &testError,
            double &classError,
            double &avg_precision,
            double &avg_recall,
            double &avg_fscore)
{
    srand(iter);
    srand48(iter);
    char train_file[1024];
    char test_file[1024];
    int pattern_dimension = mainParamList.getParam("fc_dimension").getValue().toInt();

    strcpy(train_file,selectedTrainFile.toStdString().c_str());
    strcpy(test_file,selectedTestFile.toStdString().c_str());
    int w=10;
    int total_runs=30;
    Rbf *neural = new Rbf(p->getMapper());
    neural->readPatterns(train_file);
    neural->setNumOfWeights(w);
    neural->setPatternDimension(pattern_dimension);
    testError = 0.0;
    classError  =0.0;
    avg_precision = 0.0;
    avg_recall = 0.0;
    avg_fscore = 0.0;
    for(int i=1;i<=total_runs;i++)
    {
    double d=neural->train2();
    testError+=neural->testError(test_file);
    double precision,recall,fscore;
    classError+=neural->classTestError(test_file,precision,recall);
    neural->getPrecisionRecall(test_file,precision,recall,fscore);
    avg_precision+=precision;
    avg_recall+=recall;
    avg_fscore+=fscore;
    }
    testError/=total_runs;
    classError/=total_runs;
    avg_precision/=total_runs;
    avg_recall/=total_runs;
    avg_fscore/=total_runs;
    //make red report
    red();
    printf("RBF.    TEST ERROR: %10.5lg CLASS ERROR: %10.5lg%%\n",testError,classError*100.0);
    printf("RBF.    PRECISION %10.5lg RECALL %10.5lg FSCORE %10.5lg\n",
           avg_precision,avg_recall,avg_fscore);
    reset();
    delete neural;
}

double total_neural_test_error = 0.0,total_neural_class_error=0.0,
    total_rbf_test_error=0.0,total_rbf_class_error=0.0;

double total_neural_precision=0.0,total_neural_recall=0.0,
    total_neural_fscore=0.0;


double total_rbf_precision=0.0,total_rbf_recall=0.0,
    total_rbf_fscore=0.0;

void run()
{
    loadDataFiles();
    int total_runs = mainParamList.getParam("fc_iters").getValue().toInt();
    int model_type = mainParamList.getParam("fc_model").getValue()=="rbf"?MODEL_RBF:MODEL_NEURAL;
    int pattern_dimension = mainParamList.getParam("fc_dimension").getValue().toInt();
    int pcount = mainParamList.getParam("fc_chromosomes").getValue().toInt();
    int length = mainParamList.getParam("fc_length").getValue().toInt();
    int num_weights = mainParamList.getParam("fc_weights").getValue().toInt();
    int generations = mainParamList.getParam("fc_generations").getValue().toInt();
    fc_balanceclass = mainParamList.getParam("fc_balanceclass").getValue()=="yes";
    vector<int> genome;
    genome.resize(length);
    string s;
    double best_fitness=1e+100;
    vector<int> bestgenome;
    bestgenome.resize(length);

    for(random_seed=1;random_seed<=total_runs;random_seed++)
    {
        srand(100+random_seed);
        p=new NNprogram(model_type,pattern_dimension,(char *)selectedTrainFile.toStdString().c_str());
        pop=new Population(pcount,length,p);
        pop->setLocalMethod(mainParamList.getParam("fc_local").getValue().toStdString());
        p->getModel()->setPatternDimension(pattern_dimension);
        p->getModel()->setNumOfWeights(num_weights);

        for(int i=1;i<=generations;i++)
        {
                pop->nextGeneration();
                genome=pop->getBestGenome();
                s=p->printF(genome);
                p->fitness(genome);
               // if(i%20==0)
                {
                    printf("RUN: %d GENERATION=%d FITNESS=%.8lg\nPROGRAMS=\n%s",
                        random_seed,i,pop->getBestFitness(),s.c_str());
                    printf("TEST ERROR  = %.8lg \n",
                       p->getModel()->testError((char*)selectedTestFile.toStdString().c_str()));
                    double precision,recall;
                    printf("MODEL CLASS ERROR = %.8lf%% \n",
                       p->getModel()->classTestError((char*)selectedTestFile.toStdString().c_str(),precision,recall)*100.0);
                    fflush(stdout);
                    if(fabs(pop->getBestFitness())<1e-7) break;
                }
        }

        double e1,c1,e2,c2;
        double pp1,r1,f1;
        double pp2,r2,f2;
        runNeural(random_seed,e1,c1,pp1,r1,f1);
        runRbf(random_seed,e2,c2,pp2,r2,f2);
        total_neural_test_error+=e1;
        total_neural_class_error+=c1;

        total_neural_precision+=pp1;
        total_neural_recall+=r1;
        total_neural_fscore+=f1;

        total_rbf_test_error+=e2;
        total_rbf_class_error+=c2;

        total_rbf_precision+=pp2;
        total_rbf_recall+=r2;
        total_rbf_fscore+=f2;

        if(fabs(pop->getBestFitness())<=best_fitness)
        {
            best_fitness = fabs(pop->getBestFitness());
            bestgenome=pop->getBestGenome();
        }
        if(random_seed!=total_runs)
        {
                delete p;
                delete pop;
        }
    }
    //report

    total_neural_precision/=total_runs;
    total_neural_recall/=total_runs;
    total_neural_fscore/=total_runs;


    total_rbf_precision/=total_runs;
    total_rbf_recall/=total_runs;
    total_rbf_fscore/=total_runs;

    green();
    printf("BEST FITNESS = %20.8lg\n",best_fitness);
    printf("NEURAL AVERAGES.. \tTEST: %10.5lg CLASS: %10.5lg%%\n",
          total_neural_test_error/total_runs,
          total_neural_class_error*100.0/total_runs);
    printf("PRECISION %10.5lg RECALL %10.5lg FSCORE %10.5lg \n",
           total_neural_precision,total_neural_recall,total_neural_fscore);

    printf("RBF AVERAGES..    \tTEST: %10.5lg CLASS: %10.5lg%%\n",
            total_rbf_test_error/total_runs,
            total_rbf_class_error*100.0/total_runs);
    printf("PRECISION %10.5lg RECALL %10.5lg FSCORE %10.5lg \n",
           total_rbf_precision,total_rbf_recall,total_rbf_fscore);

    genome=bestgenome;
    p->fitness(genome);
    s=p->printF(genome);
    printf("BEST FEATURES = %s \n",s.c_str());
    reset();
}

void done()
{
    if(p!=NULL)
        delete p;
    if(pop!=NULL)
        delete pop;
}

int main(int argc,char **argv)
{
    QCoreApplication app(argc,argv);
    setlocale(LC_ALL,"C");
    init();
    parseCmdLine(app.arguments());
    run();
    done();
    return 0;
}
