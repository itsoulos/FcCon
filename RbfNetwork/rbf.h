#ifndef RBF_H
#define RBF_H
# include <RbfNetwork/kmeans.h>
# include <MLMODELS/model.h>
class RbfNetwork:public Model
{
private:
    Data weight;
    vector<Data> centroid;
    Data variance;

    int   nweights;//centroids
    int   pattern_dimension;
    Data    classVector;
    void    init_arrays();
    double  gauss_function(Data &pattern,Data &center,double sigma);
    vector<Data>  matrix_transpose(vector<Data> &x);
    vector<Data>  matrix_mult(vector<Data> &x,vector<Data> &y);
    vector<Data>  matrix_inverse(vector<Data> x);
    vector<Data>  matrix_pseudoinverse(vector<Data> &x);
public:
    RbfNetwork(Mapper *m);
    void    setNumberOfWeights(int K);
    int     getNumberOfWeights();
    void    train();
    double  getOutput(Data &pattern);
    double  getClass(Data &pattern);
    double  getTrainError();
    double  getTestError();
    double  getClassError();
    double  product(Data &x,Data &y);
    virtual double train1();
    virtual double train2();
    virtual double output(Data& x);
    virtual void   getDeriv(Data &x,Data &g)
    {

    }
};

#endif // RBF_H
