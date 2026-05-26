#include "gradientdescent.h"
GradientDescent::GradientDescent(Problem *mp)
{
    myProblem = mp;
    hasInitialized  = false;
    lt=NULL;
    addParam(Parameter("gd_maxiters",200,1,10000,"Maximum iterations for gradient descent"));
    QStringList gd_linesearch;
    gd_linesearch<<"none"<<"golden"<<"fibonacci"<<"armijo";
    addParam(Parameter("gd_linesearch",gd_linesearch[0],
                       gd_linesearch,
                       "Line search. Values: none, golden, fibonacci, armijo"));
    addParam(Parameter("gd_epsilon",0.000001,0.0,1.0,"Eps value for termination"));
    addParam(Parameter("gd_rate",0.001,0.0,1.0,"Step for gradient descent"));
}


QJsonObject GradientDescent::getParams()
{
    return paramList.getParams();
}

void        GradientDescent::setParams(QJsonObject &x)
{
    paramList.setParamValuesFromJson(x);
}

void        GradientDescent::addParam(Parameter p)
{
    paramList.addParam(p);
}

void        GradientDescent::setParam(QString name,QString value,QString help)
{
    paramList.setParam(name,value,help);
}

Parameter   GradientDescent::getParam(QString name)
{
    return paramList.getParam(name);

}

 void    GradientDescent::setPoint(Data &x,double &y)
 {
     xpoint = x;
     ypoint = y;
     hasInitialized = true;
 }
void    GradientDescent::init()
{
    iteration = 0;
    maxiters = getParam("gd_maxiters").getValue().toInt();
    rate  = getParam("gd_rate").getValue().toDouble();
    eps  = getParam("gd_rate").getValue().toDouble();
    lineSearchMethod = getParam("gd_linesearch").getValue();
    if(!hasInitialized)
    {
        myProblem->getSample(xpoint);
        ypoint = myProblem->funmin(xpoint);
    }
    lt = NULL;
}

void    GradientDescent::step()
{
    iteration++;
    updaterate();
    updatepoint();
}

void    GradientDescent::showDebug()
{
    printf("GD. Iteration=%4d Optimum = %20.10lg\n",iteration,ypoint);
}
bool    GradientDescent::terminated()
{

    if(iteration>=maxiters) return true;
    Data g;
    g.resize(xpoint.size());
    myProblem->getGRMS(xpoint,g);
    double s=0.0;
    for(int i=0;i<g.size();i++)
        s+=g[i]*g[i];

    if(s<eps) return true;
    return false;
}

void    GradientDescent::updaterate()
{
    if(lineSearchMethod == "golden")
    {
        double lambda = rate;
        double a = lambda * 0.9;
        double b = lambda * 1.1;
        if(lt==NULL)
            lt = new GoldenSearch(myProblem);
        ((GoldenSearch *)lt)->setParams(a,b);

        rate = lt->getDirection(xpoint);
    }
    else
    if(lineSearchMethod == "fibonacci")
    {
        double lambda = rate;
        double a = lambda * 0.9;
        double b = lambda * 1.1;
        if(lt==NULL)
            lt = new FibonacciSearch(myProblem);
        ((FibonacciSearch *)lt)->setParams(a,b);
        rate = lt->getDirection(xpoint);
    }
    else
    if(lineSearchMethod == "armijo")
    {
        double lambda  = 2;
        if(lt == NULL)
            lt = new ArmijoSearch(myProblem);
        ((ArmijoSearch*)lt)->setLambda(lambda);

        rate = lt->getDirection(xpoint);

    }
}

void    GradientDescent::updatepoint()
{
    Data g;
    g.resize(xpoint.size());
    myProblem->granal(xpoint,g);
    unsigned int i;
    for(i=0;i<g.size();i++)
        xpoint[i]=xpoint[i]-rate * g[i];
    ypoint=myProblem->funmin(xpoint);
}

void    GradientDescent::getPoint(Data &x,double &y)
{
    x = xpoint;
    y = ypoint;
}

GradientDescent::~GradientDescent()
{
   if(lt!=NULL)
        delete lt;
}
