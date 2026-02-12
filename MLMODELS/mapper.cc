# include <stdio.h>
# include <stdlib.h>
# include <MLMODELS/mapper.h>
# include <math.h>
extern double sig(const double *x);
extern double MMin(const double *x);
extern double MMax(const double *x);
extern double MAvg(const double *x);
extern double MStd(const double *x);
Mapper::Mapper(int d)
{
	dimension = d;
	parser.resize(0);
	vars="x1";
	for(int i=1;i<dimension;i++)
	{
		vars=vars+",";
		char str[10];
		sprintf(str,"x%d",i+1);
		vars=vars+str;
	}	
}

void	Mapper::setExpr(vector<string> s)
{
	if(parser.size()!=s.size()) 
	{
	        parser.resize(s.size());
		for(int i=0;i<parser.size();i++) 
		{
			delete parser[i];
			parser[i] = new FunctionParser();
			parser[i]->AddFunction("sig",sig,1);
			/*
			parser[i]->AddFunction("xmin",MMin,1);
			parser[i]->AddFunction("xmax",MMax,1);
			parser[i]->AddFunction("xmean",MAvg,1);
			parser[i]->AddFunction("xstd",MStd,1);
			*
			*/
		}
	}
	for(int i=0;i<s.size();i++) 
	{
		int d=parser[i]->Parse(s[i],vars);
	}
}

int haveX=0;
int	Mapper::map(Matrix x,Matrix &x1)
{
	double *xx=new double[x.size()];
	for(int i=0;i<x.size();i++) xx[i]=x[i];
	int countX=0;
	extern vector<double> xcurrent;
	for(int i=0;i<parser.size();i++) 
	{
		haveX=0;
		xcurrent = x;
		x1[i]=parser[i]->Eval(xx);
		countX+=haveX==0;
		//if(!haveX) {delete[] xx;return 0;}
		if(isnan(x1[i]) || isinf(x1[i])) {delete[] xx;return 0;}
		if(parser[i]->EvalError()) {delete[] xx;return 0;}
        //if(fabs(x1[i])>1e+2) {delete[] xx;return 0;}
	}
	delete[] xx;
	haveX=countX;
	return 1;
}

Mapper::~Mapper()
{
	for(int i=0;i<parser.size();i++) delete parser[i];
}
