#ifndef FUNCTIONALRBF_H
#define FUNCTIONALRBF_H

# include <CORE/dataset.h>
# include <MLMODELS/model.h>
# include <CORE/problem.h>
# include <QJsonObject>
# include <armadillo>
#include <adept.h>

class FunctionalRbf: public Model
{
private:
    QString trainName="xy.data";
    arma::vec Linear;
    QString testName="xy.data";
    int nodes=10;
    double rbf_factor=3.0;
    int dimension=0;
    Data dclass;
    double initialLeft=-100.0;
    double initialRight= 100.0;
    int failCount=0;
    int normalTrain=0;
    double *xinput=0;
    double *yinput=0;
    vector<int> num_of_cluster_members;

    Data centers,variances;
    Data weight;
    void Kmeans(double * data_vectors,
                vector<double> &centers,
                vector<double> &variances,
                int m, int n, int K,
                vector<int>& num_of_cluster_members);
    adept::adouble aneuronOutput( vector<adept::adouble> &x, vector<double> &patt, unsigned pattDim, unsigned offset );
    adept::adouble afunmin( vector<adept::adouble> &x, vector<double> &x1 );
public:
    FunctionalRbf(Mapper *m);
    virtual double train1();
    virtual double train2();
    virtual double      output(Data& x);
    virtual Data        gradient(Data &x);
    double              neuronOutput( vector<double> &x,
                            vector<double> &patt, unsigned pattDim,
                            unsigned offset );
    arma::vec           train( vector<double> &x,bool &ok );
    double              nearestClass(double y);
    virtual void   getDeriv(Data &x,Data &g)
    {

    }
    ~FunctionalRbf();
};

#endif // FUNCTIONALRBF_H
