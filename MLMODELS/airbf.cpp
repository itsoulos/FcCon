#include "airbf.h"

AiRbf::AiRbf(Mapper *m)
    :Model(m)
{
    lambda=1e-3;
}

double  AiRbf::train1()
{
    k = num_weights;
    MatrixXd X(xpoint.size(),pattern_dimension);
    VectorXd y(xpoint.size());
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
    weight.resize(k);
    MatrixXd Phi = computePhi(X);
    VectorXd preds= Phi * weights;
    double sum = 0.0;
    for (int i = 0; i < preds.size(); i++) {
        sum+=pow(preds(i)-y(i),2.0);
    }
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
    MatrixXd centers(k, d);
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
    MatrixXd Phi(n, k + 1);
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
    centers = kmeans(X, k);
    computeSigma();
    MatrixXd Phi = computePhi(X);
    MatrixXd A = Phi.transpose() * Phi;
    A += lambda * MatrixXd::Identity(A.rows(), A.cols());
    VectorXd b = Phi.transpose() * y;
    weights = A.ldlt().solve(b);
}



AiRbf::~AiRbf()
{
    //nothing to do
}
