#include "integeranneal.h"


IntegerAnneal::IntegerAnneal(Program *pr)
{
    T0=100.0;
    neps=50;
    myProblem = pr;
}
void    IntegerAnneal::setNeps(int n)
{
    neps  = n;
}

void    IntegerAnneal::updateTemp()
{
    const double alpha = 0.8;

    T0 =T0 * alpha;
    k=k+1;
}
void    IntegerAnneal::setT0(double t)
{
    T0  = t;
}
void    IntegerAnneal::setPoint(vector<int> &g,double &y)
{
    xpoint = g;
    ypoint = y;

}
void    IntegerAnneal::getPoint(vector<int> &g,double &y)
{
    g = bestx;
    y = besty;
}

void    IntegerAnneal::Solve()
{
    bestx = xpoint;
    besty = ypoint;
    int i;
    k=1;
    vector<int> y;

    y.resize(bestx.size());
    while(true)
    {
        for(i=1;i<=neps;i++)
        {
            double fy;
            y = xpoint;
            again:
        for(int j=0;j<10;j++)
            {
            int randPos = rand() % bestx.size();
            int range = 10;
            int direction = rand() % 2==1?1:-1;
            int newValue =  y[randPos] + direction * (rand() % range);
            if(newValue<0) newValue = 0;
            y[randPos]=newValue;
        }
        fy = myProblem->fitness(y);

        if(fabs(fy)>1e+10) goto again;

        if(fabs(fy)<fabs(ypoint))
        {
            xpoint = y;
            ypoint = fy;
            if(fabs(ypoint)<fabs(besty))
            {
                bestx = xpoint;
                besty = ypoint;
            }
        }
        else
        {
            double r = fabs((rand()*1.0)/RAND_MAX);
            double ratio = exp(-(fy-ypoint)/T0);
            double xmin = ratio<1?ratio:1;
            if(r<xmin)
            {
                xpoint = y;
                ypoint = fy;
                if(fabs(ypoint)<fabs(besty))
                {
                    bestx = xpoint;
                    besty = ypoint;
                }
            }
        }
        }
        updateTemp();
        if(T0<=1e-6) break;
       printf("Iteration: %4d Temperature: %20.10lg Value: %20.10lg\n",
               k,T0,besty);

    }
}

IntegerAnneal::~IntegerAnneal()
{
    //nothing here
}
