# include <MLMODELS/rbf_model.h>

Rbf::Rbf(Mapper *m)
	:Model(m)
{
	centers = NULL;
	variances = NULL;
	weights = NULL;
	input  = NULL;
	weight.resize(0);
}

double Rbf::train1()
{
    int noutput=1;

	if(weight.size() != noutput * num_weights)
	{
		weight.resize(num_weights*noutput);
		setDimension(num_weights*noutput);
		if(centers)
		{
		delete[] centers;
		delete[] variances;
		delete[] weights;
		delete[] input;
		}
		centers = new double[num_weights * pattern_dimension];
		variances = new double[num_weights * pattern_dimension];
		weights = new double[num_weights*noutput];
		input = new double[pattern_dimension*xpoint.size()];
        Output=new double[noutput * xpoint.size()];

	}

    if(!mapTrainSet())
    {
        return 1e+100;
    }

    for(int i=0;i<(int)xpoint.size();i++)
    {
		for(int j=0;j<pattern_dimension;j++)
		{
			
			input[i*pattern_dimension+j]=xpoint[i][j];
            if(fabs(xpoint[i][j])>=1e+10 || isnan(xpoint[i][j]) || isinf(xpoint[i][j]))
            {
                return 1e+100;
            }
		}
		Output[i]=ypoint[i];
    }
    Kmeans(input,centers,variances,
			xpoint.size(),pattern_dimension,num_weights);
	
    int icode=train_rbf(pattern_dimension,num_weights,noutput,xpoint.size(),
			centers,variances,weights,input,Output);
	double v =0.0;
	v=funmin(weight);
	if(icode==1) return 1e+100;
	return v;
}

int maxWeight = 50;
double	Rbf::setWeightValuesFromPattern(double *pattern,int size)
{
	int countPattern=0;
	int noutput=1;
	for(int i=0;i<size/2;i++)
	{
		if(fabs(pattern[i])>0.5) countPattern++;
	}
	if(countPattern==0) return 1e+100;
	num_weights = countPattern;
	
	if(centers==NULL)
	{
		centers = new double[maxWeight * pattern_dimension];
		variances = new double[maxWeight * pattern_dimension];
		weights = new double[maxWeight*noutput];
		input = new double[pattern_dimension*xpoint.size()];
	Output=new double[noutput * xpoint.size()];
	}

	if(weight.size() != noutput * num_weights)
	{
		weight.resize(num_weights*noutput);
		setDimension(num_weights*noutput);
	}
	int icount=0;
	for(int i=0;i<size/2;i++)
	{
		if(fabs(pattern[i])>0.5) 
			weights[icount++]=pattern[2*i];
	}

    if(!mapTrainSet()) return 1e+100;

	for(int i=0;i<xpoint.size();i++) 
	{
		for(int j=0;j<pattern_dimension;j++)
		{		
			input[i*pattern_dimension+j]=xpoint[i][j];
			if(fabs(xpoint[i][j])>=1e+10 || isnan(xpoint[i][j]) || isinf(xpoint[i][j]))
			{
				return 1e+100;
			}
		}
        Output[i]=ypoint[i];
	}
	srand48(1);
        Kmeans(input,centers,variances,
			xpoint.size(),pattern_dimension,num_weights);
	
        int icode=train_rbf(pattern_dimension,num_weights,noutput,xpoint.size(),
     			centers,variances,weights,input,Output);
	double v =0.0;

	v=funmin(weight);
	return v;
}

double Rbf::train2()
{
	return train1();
    /*double v;
	double pattern[2 * maxWeight];
	RbfSolve(this,pattern,v,0,0);
	setWeightValuesFromPattern(pattern, 2* maxWeight);
    return -v;//return train1();*/
}

double Rbf::output(Data &x)
{
	if(x.size()==0) return 1e+100;
    double v[1];
	double *xt=new double[x.size()];
	double penalty=0.0;
	for(int i=0;i<x.size();i++) 
	{
		xt[i]=x[i];
	}
    create_rbf(pattern_dimension,num_weights,1,
			centers,variances,weights,xt,v);
	delete[] xt;
	return v[0];
}

void   Rbf::getDeriv(Data &x,Data &g)
{
}

Rbf::~Rbf()
{

	delete[] centers;
	delete[] variances;
	delete[] weights;
	delete[] input;
    delete[] Output;
}
