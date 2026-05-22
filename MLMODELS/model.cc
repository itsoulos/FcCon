# include <MLMODELS/model.h>
# include <stdlib.h>
# include <string.h>
# include <stdio.h>
# include <math.h>

double Model::distance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        sum += std::pow(a[i] - b[i], 2);
    }
    return std::sqrt(sum);
}

static int nearestClassIndex(vector<double> &dclass,double value)
{
    int pos=-1;
    double dmin=1e+100;
    for(int i=0;i<dclass.size();i++)
    {
        double d = fabs(dclass[i]-value);
        if(d<dmin)
        {
            dmin = d;
            pos = i;
        }
    }
    return pos;
}



Model::Model(Mapper *m)
{
	num_weights = 1;
	pattern_dimension = 0;
	mapper = m;
	isvalidation=0;
}

Data    	Model::getWeights()
{
	return weight;
}


int	Model::getOriginalDimension() const
{
	return original_dimension;
}


int	Model::getNumPatterns() const
{
	return ypoint.size();
}

void        Model::randomizeWeights()
{
	weight.resize((pattern_dimension+2)*num_weights);
	setDimension(weight.size());
	for(int i=0;i<weight.size();i++) weight[i]=0.1*(2.0*drand48()-1.0);
}

void    	Model::setPatternDimension(int d)
{
    if(pattern_dimension!=d || xpoint.size()==0)
    {
        xpoint.resize(trainSet->count());
		pattern_dimension = d;
        for(int i=0;i<(int)xpoint.size();i++)
			xpoint[i].resize(pattern_dimension);
        ypoint.resize(xpoint.size());
        ypoint=trainSet->getAllYPoints();
	}
}


bool    Model::mapTrainSet()
{

    for(int i=0;i<(int)xall.size();i++)
    {
        if(!mapper->map(xall[i],xpoint[i])) return false;
    }
    return true;
}

void    Model::setTrainSet(Dataset *t)
{
    extern bool fc_enablesmote;
    extern bool fc_enablenorm;
    trainSet = t;
    if(fc_enablesmote)
        trainSet->makeSmote();
    if(fc_enablenorm)
        trainSet->normalizeMean();
        //trainSet->normalizeMinMax();
    xall = trainSet->getAllXpoint();
}

void    Model::setTestSet(Dataset *t)
{
    extern bool fc_enablenorm;
    testSet = t;
    if(fc_enablenorm)
        testSet->normalizeMean();
        //testSet->normalizeMinMax();
}

void	Model::setNumOfWeights(int w)
{
	num_weights = w;
}

Data    	Model::getXpoint(int pos)
{
	return xpoint[pos];
}

double  Model::getYPoint(int pos)
{
	return ypoint[pos];
}

double  Model::getModelAtPoint(int pos)
{
	double v = output(xpoint[pos]);
	return v;
}

int	Model::getPatternDimension() const
{
	return pattern_dimension;
}

double	Model::valError()
{
	double s=0.0;
	for(int i=4*xpoint.size()/5;i<xpoint.size();i++)
	{
		double v = output(xpoint[i]);
		s+=(v-ypoint[i])*(v-ypoint[i]);
	}
	return s;
}

double  Model::getAverageClassError(Data &x)
{
    if(weight.size()!=x.size()) weight.resize(x.size());
    for(int i=0;i<x.size();i++) weight[i] = x[i];
    double sum = 0.0;
    int end=xpoint.size();
    if(isvalidation) end=4*xpoint.size()/5;
    vector<int> missed,belong;
    vector<double> dclass = trainSet->getPatternClass();
    missed.resize(dclass.size());
    belong.resize(dclass.size());
    double dsum = 0.0;
    for(int i=0;i<(int)missed.size();i++)
    {
        missed[i]=0;
        belong[i]=0;
    }
    for(int i=0;i<end;i++)
    {
        double v = output(xpoint[i]);
        double c1 = trainSet->nearestClassIndex(v);
        double c2 = trainSet->nearestClassIndex(ypoint[i]);
        if(fabs(c1-c2)>1e-5)
        {
            missed[(int)c2]++;
            dsum+=1.0;
        }
        belong[(int)c2]++;
    }

    for(int i=0;i<(int)missed.size();i++)
    {
        double dc = missed[i]*100.0/belong[i];
        sum+=dc;
    }
    extern bool fc_enableclassfitness;
    if(fc_enableclassfitness)
        return dsum*100.0/ypoint.size();
    return sum/dclass.size();
}

double	Model::funmin(Data &x)
{
    extern bool fc_balanceclass;
    extern bool fc_enablemean;
    extern bool fc_enableclassfitness;
    if(fc_balanceclass || fc_enableclassfitness)
        return getAverageClassError(x);

    if(weight.size()!=x.size())
        weight.resize(x.size());
    for(int i=0;i<(int)x.size();i++)
        weight[i] = x[i];


    if(fc_enablemean)
    {

        double avg_precision,avg_recall,avg_fscore;
        getPrecisionRecall(
            trainSet,
            avg_precision,
            avg_recall,
            avg_fscore);

        double d = 100.0*(1.0-sqrt(avg_precision * avg_recall));

        return d;
    }
	double s=0.0;
	int end=xpoint.size();
	if(isvalidation) end=4*xpoint.size()/5;
	for(int i=0;i<end;i++)
	{
		double v = output(xpoint[i]);
		double e=v-ypoint[i];
        e=e*e;
        if(std::isnan(v) || std::isinf(v)) return 1e+100;
		s+=e;
		if(isnan(s) || isinf(s)) return 1e+100;
	}
	return s;
}

void  Model::granal(Data &x,Data &g)
{
	if(weight.size()!=x.size())
	weight.resize(x.size());
    for(int i=0;i<(int)x.size();i++)
	{
		weight[i] = x[i];
		g[i]=0.0;
	}
	double s=0.0;
    Data gtemp;
	gtemp.resize(g.size());
	int end=xpoint.size();
	if(isvalidation) end=4*xpoint.size()/5;
	for(int i=0;i<end;i++)
	{
        double	e=output(xpoint[i])-ypoint[i];
		getDeriv(xpoint[i],gtemp);
        for(int j=0;j<(int)g.size();j++)
		{
			g[j]+=2.0*e*gtemp[j];
		}
	}
}


double	Model::testError()
{
	double testy;
    Data testx;
    testx.resize(pattern_dimension);
    Data xx;
    xx.resize(testSet->dimension());
    double sum = 0.0;

    for(int i=0;i<testSet->count();i++)
    {
        xx =testSet->getXPoint(i);
        mapper->map(xx,testx);
		double d=output(testx);
        testy=testSet->getYPoint(i);
		sum+=pow(d-testy,2.0);
	}

	return (sum);
}

double	Model::classTestError()
{

    Data testx;
    testx.resize(pattern_dimension);
    Data xx;
    xx.resize(testSet->dimension());
	double sum = 0.0;
    vector<double> dclass = testSet->getPatternClass();
    vector<int> missed,belong;
    missed.resize(dclass.size());
    belong.resize(dclass.size());
    for(int i=0;i<(int)missed.size();i++)
    {
        missed[i]=0;
        belong[i]=0;
    }
    for(int i=0;i<testSet->count();i++)
	{
        xx =testSet->getXPoint(i);
        mapper->map(xx,testx);
        double d=output(testx);
        double y = testSet->getYPoint(i);
        int c1 = testSet->nearestClassIndex(d);
        int c2 = testSet->nearestClassIndex(y);
        if(c1!=c2)
        {
            missed[c2]++;
        }
        belong[c2]++;
        sum+=(c1!=c2);
	}
    printf("TEST REPORT\n");
    for(int i=0;i<(int)dclass.size();i++)
    {
        if(belong[i]==0)
            printf("Error on class %lf \n",dclass[i]);
        printf("ERROR CLASS[%d]=%.2lf%%\n",i,missed[i]*100.0/belong[i]);
    }
    return (sum*100.0)/testSet->count();
}


void        Model::printConfusionMatrix(
                                 vector<double> &dclass,
                                 vector<double> &T,vector<double> &O,
                                 vector<double> &precision,
                                 vector<double> &recall)
{
    int i,j;
    int N=T.size();
    int nclass=dclass.size();
    precision.resize(nclass);
    recall.resize(nclass);
    int **CM;
    //printf("** CONFUSION MATRIX ** Number of classes: %d\n",nclass);
    CM=new int*[nclass];
    for(i=0;i<nclass;i++) CM[i]=new int[nclass];
    for(i=0;i<nclass;i++)
        for(j=0;j<nclass;j++) CM[i][j] = 0;

    for(i=0;i<N;i++) CM[(int)T[i]][(int)O[i]]++;
    for(i=0;i<nclass;i++)
    {
        double sum = 0.0;
        for(j=0;j<nclass;j++)
            sum+=CM[j][i];
        precision[i]=sum==0?-1:CM[i][i]/sum;
        sum = 0.0;
        for(j=0;j<nclass;j++)
            sum+=CM[i][j];
        recall[i]=sum==0?-1:CM[i][i]/sum;
    }
    for(i=0;i<nclass;i++)
    {
        for(j=0;j<nclass;j++)
        {
            //printf("%4d ",CM[i][j]);
        }
        //printf("\n");
        delete[] CM[i];
    }
    delete[] CM;
}

void    Model::getPrecisionRecall(
                               Dataset *t,
                               double &avg_precision,
                               double &avg_recall,
                               double &avg_fscore)
{

    int count=t->count();
    vector<double> dclass=t->getPatternClass();
    vector<vector<double>> testx=t->getAllXpoint();
    vector<double>  testy=t->getAllYpoint();

    vector<double> T;
    vector<double> O;
    T.resize(count);
    O.resize(count);

    vector<double> xx;
    xx.resize(pattern_dimension);

    for(int i=0;i<count;i++)
    {
        mapper->map(testx[i],xx);
        double tempOut = output(xx);
        T[i]=t->nearestClassIndex(testy[i]);
        O[i]=t->nearestClassIndex(tempOut);
    }

    vector<double> precision;
    vector<double> recall;
    vector<double> fscore;
    fscore.resize(dclass.size());
    avg_precision = 0.0, avg_recall = 0.0,avg_fscore=0.0;
    printConfusionMatrix(dclass,T,O,precision,recall);
    int icount1=dclass.size();
    int icount2=dclass.size();
    for(int i=0;i<(int)dclass.size();i++)
    {
        if(precision[i]>=0)
            avg_precision+=precision[i];
        else icount1--;
        if(recall[i]>=0)
            avg_recall+=recall[i];
        else icount2--;
        fscore[i]=2.0*precision[i]*recall[i]/(precision[i]+recall[i]);
        avg_fscore+=fscore[i];
    }
    avg_precision/=icount1;
    avg_recall/=icount2;
    avg_fscore=2.0 * avg_precision * avg_recall/(avg_precision+avg_recall);

}

void        Model::enableValidation()
{
	isvalidation=1;
}

Model::~Model()
{
}
