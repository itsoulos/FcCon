#include "armijosearch.h"

ArmijoSearch::ArmijoSearch(Problem *p)
    :LineSearch(p)
{

}

void    ArmijoSearch::setLambda(double l)
{
    ArmijoLambda = l;
}

double  ArmijoSearch::getDirection(Data &x)
{
    double f0 = myProblem->funmin(x);
    double beta =0.001;
    double t = 0.5;
    int iteration = 0;
      double s=0.0;
    Data g;
    g.resize(x.size());
    do
    {
            ArmijoLambda = ArmijoLambda * t;
            iteration++;
            if(iteration>=20) break;

            s=myProblem->getGRMS(x,g);
            }while(fl(x,ArmijoLambda)>f0-ArmijoLambda*beta*s);
    return ArmijoLambda;
}
