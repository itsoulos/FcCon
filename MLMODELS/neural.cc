# include <MLMODELS/neural.h>
# include <math.h>
# include <MLMODELS/gensolver.h>
int pass=0;
double maxx=-1e+100;
double 	lmderFunmin(double *par,int x,void *fdata);
void	lmderGradient(double *g,double *par,int x,void *fdata);
static double sig(double x)
{
	return 1.0/(1.0+exp(-x));
}

static double sigder(double x)
{
	double s=sig(x);
	return s*(1-s);
}

Neural::Neural(Mapper *m):Model(m)
{
}

void	Neural::setWeights(Matrix x)
{
	for(int i=0;i<weight.size();i++) weight[i]=x[i];
}

double Neural::train1()
{
	for(int i=0;i<xpoint.size();i++) 
	{
		int d=mapper->map(origx[i],xpoint[i]);
		if(!d) return 1e+100;
		for(int j=0;j<pattern_dimension;j++)
		{
			if(isinf(xpoint[i][j])) return 1e+100;
		}
	}
	double v;
	MinInfo Info;
	Info.p = this;
	Info.iters=200;
	return tolmin(weight,Info);

	double oldval=funmin(weight);
	Matrix saveweight;
	saveweight.resize(weight.size());
	saveweight=weight;
	for(int i=1;i<=200;i++)
	{
		v=tolmin(weight,Info);
		v=v*(1.0+100.0*countViolate(10.0));
		if(v>oldval)
		{
			weight=saveweight;
			v=oldval;
			break;
		}
		oldval=v;
		saveweight=weight;
	}
	v=funmin(weight);
	return v;
}

double	Neural::countViolate(double limit)
{
	//metraei posoi komboi prokaloyn problima sta sigmoeidi
	int count = 0;
	int nodes=weight.size() / (pattern_dimension + 2);
	for(int i=0;i<xpoint.size();i++)
	{
		for(int n=1;n<=nodes;n++)
		{
              		double arg = 0.0;
              		for(int j=1;j<=pattern_dimension;j++)
              		{
                 	 int pos=(pattern_dimension+2)*n-(pattern_dimension+1)+j-1;
                  	 arg+=weight[pos]*xpoint[i][j-1];
			}
              	arg+=weight[(pattern_dimension+2)*n-1];
		count+=(fabs(arg)>=limit);
	      }
	}
	return count*1.0/(xpoint.size()*nodes);
}


void devmarFunmin(double *p,double *x,int m,int n,void *data)
{
	Neural *neural = (Neural *)data;
	Matrix w = neural->getWeights();
	for(int i=0;i<m;i++) w[i]=p[i];
	neural->setWeights(w);	
	for(int i=0;i<n;i++)
	{
		double v=neural->getModelAtPoint(i);	
		x[i]=pow(v-neural->getYPoint(i),2.0);
	}
}

void devmarJac(double *p, double *jac, int m, int n, void *data)
{
	Neural *neural = (Neural *)data;
	Matrix w = neural->getWeights();
	for(int i=0;i<m;i++) w[i]=p[i];
	neural->setWeights(w);	
	for(int i=0;i<n;i++)
	{
	}
}


double Neural::train2()
{
	double v;
	for(int i=0;i<xpoint.size();i++) 
	{
		mapper->map(origx[i],xpoint[i]);
	}
/*	randomizeWeights();
	LMstat lmstat;
	levmarq_init(&lmstat);
	int n_iterations;
	int N_PARAMS = weight.size();
	int N_M = ypoint.size();
	double t_data[N_M];
	double params[weight.size()];
	for(int i=0;i<weight.size();i++) params[i]=weight[i];
	for(int i=0;i<ypoint.size();i++) t_data[i]=ypoint[i];
double opts[LM_OPTS_SZ], info[LM_INFO_SZ];

int ret=	dlevmar_dif(devmarFunmin,params,t_data,N_PARAMS,N_M,1000,opts,info,NULL,NULL,this);
 printf("Levenberg-Marquardt returned %d in %g iter, reason %g\nSolution: ", ret, info[5], info[6]);
  for(int i=0; i<N_PARAMS; ++i)
    printf("%.7g ", params[i]);
  printf("\n");
v=info[6];
*/	
	MinInfo Info;
	Info.p=this;
	Info.iters=2001;
	randomizeWeights();
	lmargin.resize(weight.size());
	rmargin.resize(weight.size());
	for(int i=0;i<weight.size();i++)
	{
		lmargin[i]=-10.0;
		rmargin[i]= 10.0;
	}
//	GenSolve(this,weight,v,0,0);

	v=tolmin(weight,Info);
	return v;
}


double Neural::output(Matrix x)
{
	double arg=0.0,per=0.0;
	int dimension = pattern_dimension;
	int nodes=weight.size()/(pattern_dimension+2);
        for(int i=1;i<=nodes;i++)
        {
              arg = 0.0;
              for(int j=1;j<=dimension;j++)
              {
                  int pos=(dimension+2)*i-(dimension+1)+j-1;
                  arg+=weight[pos]*x[j-1];
              }
	      
              arg+=weight[(dimension+2)*i-1];
              per+=weight[(dimension+2)*i-(dimension+1)-1]*sig(arg);
        }
        return per;
}

void   Neural::getDeriv(Matrix x,Matrix &g)
{
  	double arg;
        double s;
        int nodes=weight.size()/(pattern_dimension+2);
	int dimension = pattern_dimension;
        double f,f2;
        for(int i=1;i<=nodes;i++)
        {
                arg = 0.0;
                for(int j=1;j<=dimension;j++)
                {
                  	int pos=(dimension+2)*i-(dimension+1)+j-1;
                        arg+=weight[pos]*x[j-1];
                }
                arg+=weight[(dimension+2)*i-1];
                f=sig(arg);
		f2=f*(1.0 -f );
                g[(dimension+2)*i-1]=weight[(dimension+2)*i-(dimension+1)-1]*f2;
                g[(dimension+2)*i-(dimension+1)-1]=f;
                for(int k=1;k<=dimension;k++)
                {
                        g[(dimension+2)*i-(dimension+1)+k-1]=
                             x[k-1]*f2*weight[(dimension+2)*i-(dimension+1)-1];
                }
        }
}


Neural::~Neural()
{
}
