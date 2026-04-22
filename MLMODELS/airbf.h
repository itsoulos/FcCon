#ifndef AIRBF_H
#define AIRBF_H
# include <MLMODELS/model.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <Eigen/Dense>
# include <stdio.h>
using namespace std;
using namespace Eigen;
class AiRbf :public Model
{
private:
    int k;
    double lambda;
    double sigma;
    MatrixXd centers;
    VectorXd weights;
public:
    AiRbf(Mapper *m);
    virtual double train1();
    virtual double train2();
    virtual double output(Data& x);
    double  gaussian(double d, double sigma);
    MatrixXd    kmeans(const MatrixXd &X, int k, int iters = 100);
    void    computeSigma();
    MatrixXd    computePhi(const MatrixXd &X);
    void    train(const MatrixXd &X, const VectorXd &y);
    virtual void   getDeriv(Data &x,Data &g)
    {

    }
    ~AiRbf();
};

#endif // AIRBF_H
