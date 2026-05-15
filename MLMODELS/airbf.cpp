#include "airbf.h"

AiRbf::AiRbf(Mapper *m)
    :Model(m)
{
    lambda=1e-3;
}

double  AiRbf::train1()
{
    extern bool fc_balanceclass;
    extern bool fc_enablemean;
    extern bool fc_enableclassfitness;
    k = num_weights;
    if((int)weight.size()!=k)
    {
        weight.resize(k);
        X.resize(xpoint.size(),pattern_dimension);
        y.resize(xpoint.size());
    }
    if(!mapTrainSet()) return 1e+100;
    for(int i=0;i<(int)xpoint.size();i++)
    {
        for(int j=0;j<pattern_dimension;j++)
        {

            X(i,j)=xpoint[i][j];
        }
        y(i)=ypoint[i];
    }
    train(X,y);
    computePhi(X);
    VectorXd preds= Phi * weights;
    double sum = 0.0;
    vector<int> missed,belong;
    vector<double> dclass = trainSet->getPatternClass();
    missed.resize(dclass.size());
    belong.resize(dclass.size());
    double dsum = 0.0;
    for(int i=0;i<(int)missed.size();i++)
    {
        missed[i]=0;
        belong[i]=0;
    }
    for (int i = 0; i < preds.size(); i++) {
        if(isnan(preds(i)) || isinf(preds(i))) return 1e+100;
        sum+=pow(preds(i)-y(i),2.0);
        int c1 = trainSet->nearestClassIndex(preds(i));
        int c2 = trainSet->nearestClassIndex(y(i));
       // printf("INDEXES %d %d %lf \n",c1,c2,preds(i));
        if(fabs(c1-c2)>1e-5)
        {
            missed[(int)c2]++;
            dsum+=1.0;
        }
        belong[(int)c2]++;
    }
    double sum2=0.0;
    for(int i=0;i<(int)missed.size();i++)
    {
        double dc = missed[i]*100.0/belong[i];
        sum2+=dc;
    }
    sum2=sum2/dclass.size();
    if(fc_balanceclass) return sum2;
    if(fc_enableclassfitness) return dsum*100.0/ypoint.size();
    return sum;
}

double  AiRbf::train2()
{
    return train1();
}

double  AiRbf::output(Data& x)
{
    if (x.size() != centers.cols()) {
        cerr << "Dimension mismatch!" << endl;
        return 0.0;
    }

    VectorXd phi(k + 1);
    phi(0) = 1.0; // bias
    for (int j = 0; j < k; j++) {
        // convert center row to VectorXd
    VectorXd c = centers.row(j).transpose();
    VectorXd x_eig(x.size());
        for (size_t i = 0; i < x.size(); i++)
            x_eig(i) = x[i];
        double d = (x_eig - c).norm();
        phi(j + 1) = gaussian(d, sigma);
    }
    return weights.dot(phi);
}

double  AiRbf::gaussian(double d, double sigma)
{
    return exp(-(d * d) / (2 * sigma * sigma));
}

MatrixXd    AiRbf::kmeans(const MatrixXd &X, int k, int iters)
{
    int n = X.rows(), d = X.cols();
    mt19937 gen(1);
    uniform_int_distribution<> dis(0, n - 1);
    if(centers.rows()==0)
    {
        centers.resize(k,d);
    }
    for (int i = 0; i < k; i++)
        centers.row(i) = X.row(dis(gen));

    VectorXi labels(n);

    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < n; i++) {
            double best = 1e18;
            int bestIdx = 0;

            for (int j = 0; j < k; j++) {
                double d_ = (X.row(i) - centers.row(j)).norm();
                if (d_ < best) {
                    best = d_;
                    bestIdx = j;
                }
            }
            labels(i) = bestIdx;
        }
        MatrixXd newCenters = MatrixXd::Zero(k, d);
        VectorXi counts = VectorXi::Zero(k);
        for (int i = 0; i < n; i++) {
            newCenters.row(labels(i)) += X.row(i);
            counts(labels(i))++;
        }
        for (int j = 0; j < k; j++) {
            if (counts(j) > 0)
                newCenters.row(j) /= counts(j);
        }
        centers = newCenters;
    }
    return centers;

}

void    AiRbf::computeSigma()
{
    double sum = 0.0;
    int count = 0;
    for (int i = 0; i < k; i++)
        for (int j = i + 1; j < k; j++) {
            sum += (centers.row(i) - centers.row(j)).norm();
            count++;
        }
    sigma = sum / count;
}

MatrixXd  AiRbf::computePhi(const MatrixXd &X)
{
    int n = X.rows();
    if(Phi.rows()!=n)
    {
        Phi.resize(n,k+1);
    }
    for (int i = 0; i < n; i++) {
        Phi(i, 0) = 1.0;

        for (int j = 0; j < k; j++) {
            double d = (X.row(i) - centers.row(j)).norm();
            Phi(i, j + 1) = gaussian(d, sigma);
        }
    }
    return Phi;
}

void    AiRbf::train(const MatrixXd &X, const VectorXd &y)
{

    kmeans(X, k);
    computeSigma();
    computePhi(X);
    MatrixXd A = Phi.transpose() * Phi;
    A += lambda * MatrixXd::Identity(A.rows(), A.cols());
    VectorXd b = Phi.transpose() * y;
    weights = A.ldlt().solve(b);
}



AiRbf::~AiRbf()
{
    //nothing to do
}
