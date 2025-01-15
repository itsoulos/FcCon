# include <MLMODELS/rbf_model.h>
//# define CLASS

Rbf::Rbf(Mapper *m)
	:Model(m)
{
	centers = NULL;
	variances = NULL;
	weights = NULL;
	input  = NULL;
	weight.resize(0);
}

double *Output;

double Rbf::train1()
{
	int noutput=1;
	#ifdef CLASS
		noutput=2;
	#endif
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
	}
	Output=new double[noutput * xpoint.size()];

	Matrix xx;
	xx.resize(original_dimension);
	for(int i=0;i<xpoint.size();i++) 
	{
		int d=mapper->map(origx[i],xpoint[i]);
		if(!d) 
		{
			delete[] Output;
			return 1e+100;
		}
		for(int j=0;j<pattern_dimension;j++)
		{
			
			input[i*pattern_dimension+j]=xpoint[i][j];
		}
		if(!d)  
		{
			printf("fail3\n");
			delete[] Output;
			return 1e+100;
		}
#ifndef CLASS
		Output[i]=ypoint[i];
#endif
	}
#ifdef CLASS
	for(int i=0;i<ypoint.size();i++)
	{
		Output[i*2+0]=ypoint[i]>0?0:1;
		Output[i*2+1]=ypoint[i]>0?1:0;
	}
#endif


        Kmeans(input,centers,variances,
			xpoint.size(),pattern_dimension,num_weights);
	
        int icode=train_rbf(pattern_dimension,num_weights,noutput,xpoint.size(),
			centers,variances,weights,input,Output);
	double v =0.0;
#ifndef CLASS
	double norm=0.0;
	for(int i=0;i<weight.size();i++) {weight[i]=weights[i];norm+=fabs(weight[i]);}
	v=funmin(weight);
#else
	for(int i=0;i<ypoint.size();i++)
	{
		double pattern[2];
		if(ypoint[i]>0) {pattern[0]=0;pattern[1]=1;}
		else            {pattern[0]=1;pattern[1]=0;}
		
		double outv[noutput];
		double *xt=new double[pattern_dimension];
		for(int j=0;j<pattern_dimension;j++) xt[j]=xpoint[i][j];
		create_rbf(pattern_dimension,num_weights,noutput,
				centers,variances,weights,xt,outv);
		v+=pow(outv[0]-pattern[0],2.0)+pow(outv[1]-pattern[1],2.0);
		delete[] xt;
	}
#endif
	delete[] Output;
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

	Matrix xx;
	xx.resize(original_dimension);
	for(int i=0;i<xpoint.size();i++) 
	{
		int d=mapper->map(origx[i],xpoint[i]);
		if(!d) 
		{
			return 1e+100;
		}
		for(int j=0;j<pattern_dimension;j++)
		{
			
			input[i*pattern_dimension+j]=xpoint[i][j];
			if(fabs(xpoint[i][j])>=1e+10 || isnan(xpoint[i][j]) || isinf(xpoint[i][j]))
			{
				return 1e+100;
			}
		}
		if(!d)  
		{
			return 1e+100;
		}
		Output[i]=ypoint[i];
	}



	srand48(1);
        Kmeans(input,centers,variances,
			xpoint.size(),pattern_dimension,num_weights);
	
        int icode=train_rbf(pattern_dimension,num_weights,noutput,xpoint.size(),
     			centers,variances,weights,input,Output);
	double v =0.0;
	double norm=0.0;
	for(int i=0;i<weight.size();i++) {weight[i]=weights[i];norm+=fabs(weight[i]);}
	
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

double Rbf::output(Matrix x)
{
	if(x.size()==0) return 1e+100;
	int noutput=1;
#ifdef CLASS
	noutput=2;
#endif
	double v[noutput];
	double *xt=new double[x.size()];
	double penalty=0.0;
	for(int i=0;i<x.size();i++) 
	{
		xt[i]=x[i];
	}
	create_rbf(pattern_dimension,num_weights,noutput,
			centers,variances,weights,xt,v);
	delete[] xt;
#ifndef CLASS
	return v[0];
#else
	return (v[0]>v[1]?0:1);	
#endif
}

void   Rbf::getDeriv(Matrix x,Matrix &g)
{
}

Rbf::~Rbf()
{

	delete[] centers;
	delete[] variances;
	delete[] weights;
	delete[] input;

}
