#include "functionalrbf.h"

double FunctionalRbf::train1()
{
    extern bool fc_balanceclass;
    extern bool fc_enablemean;
    extern bool fc_enableclassfitness;
    if(xinput == NULL)
    {
        xinput=new double[ xpoint.size() * xpoint[0].size()];
        yinput=new double[ xpoint.size()];
    }
    if(!mapTrainSet()) return 1e+100;
    nodes = num_weights;

    int icount=0;
    for(int i=0;i<(int)xpoint.size();i++)
    {
        for(int j=0;j<(int)xpoint[0].size();j++)
            xinput[icount++]=xpoint[i][j];
        yinput[i]=ypoint[i];
        bool found=false;
        for(int j=0;j<(int)dclass.size();j++)
        {
            if(fabs(dclass[j]-ypoint[i])<1e-7)
            {
                found=true;
                break;
            }

        }
        if(!found)
            dclass.push_back(ypoint[i]);
    }
    centers.resize(nodes * xpoint[0].size());
    variances.resize(nodes * xpoint[0].size());
    Kmeans(xinput,centers,variances,xpoint.size(),xpoint[0].size(),
           nodes,
           num_of_cluster_members);
    bool ok;

    weight.resize(centers.size()+variances.size()+nodes);
    icount = 0;
    for(int i=0;i<centers.size();i++)
        weight[icount++]=centers[i];
    for(int i=0;i<variances.size();i++)
        weight[icount++]=variances[i];
    double errorSum=0.0;
    Linear = train(weight,ok);
    if(!ok) return 1e+100;

    for(unsigned i = 0; i < xpoint.size(); i++){
        Data pattern = xpoint[i];
        arma::vec neuronOuts(nodes);
        for(unsigned j = 0; j < nodes;j++){
            neuronOuts[j] = neuronOutput(weight,pattern,pattern.size(),j);
        }
        double tempOut = arma::dot(neuronOuts,Linear);
        errorSum += ( tempOut - ypoint[i] ) * ( tempOut - ypoint[i] );
    }
    if(fabs(errorSum)>1e+8) return 1e+100;
    for(int i=0;i<nodes;i++)
    {
        weight[icount++]=Linear(i);
    }
    return errorSum;
}

double  FunctionalRbf::train2()
{
    return train1();
}

void FunctionalRbf::Kmeans(double * data_vectors,
            vector<double> &centers,
            vector<double> &variances,
            int m, int n, int K,
            vector<int>& num_of_cluster_members)
{
    int i=0;
    int j=0;
    int l=0;
    int k=0;
    double * new_centers = (double*)malloc(sizeof(double)*K*n);
    int **cluster_members=new int*[K];
    for(int i=0;i<K;i++)
        cluster_members[i]=new int[m];
    num_of_cluster_members.resize(K);

    double distance=0;
    double total_distance=0;
    double min_distance=0;
    int min_center=0;
    int match=0;
    int cur_match=0;
    int new_cen=0;
    int *random_centers=new int[K];
    int found=1;
    int iterations=0;


    // Assign a random center to each example in the training set
    for(i=0; i<K; i++)
    {
        do{
            random_centers[i]=(int)((m-1) * rand()*1.0/RAND_MAX);
        }while(random_centers[i]>m);

    }
    // Search the dataset and assign duplicate examples to different centers
    for(i=0; i<K; i++)
        for(j=0; j<K; j++)
            if(i!=j)
                if(random_centers[i] == random_centers[j]) {
                    for(l=0; l<m; l++) {
                        found=1;
                        for(k=0; k<K; k++) {
                            if(l==random_centers[k])
                                found=0;
                        }
                        if(found==1) {
                            new_cen=l;
                            break;
                        }
                    }
                    random_centers[j]=new_cen;
                }


    // Create the initial random centers
    for(i=0; i<K; i++) {
        for(j=0; j<n; j++) {
            //if(random_centers[i]==m)
            centers[i*n + j] = data_vectors[random_centers[i]*n + j];
            new_centers[i*n + j] = 0;
            variances[i*n + j] = 0;
        }
        num_of_cluster_members[i]=0;
        for(j=0; j<m; j++)
            cluster_members[i][j]=0;
    }

    // Main K-Means loop starts here
    iterations=0;
    while(1) {

        /* Loop over all points in the dataset */
        for(i=0; i<m; i++) {

            /* Estimate the closest center to point i */
            for(j=0; j<K; j++) {
                distance=0;
                for(l=0; l<n; l++) {
                    distance += pow((data_vectors[i*n + l] - centers[j*n + l]),2.0);
                }

                if(j==0) {
                    min_distance = distance;
                    min_center = j;
                    continue;
                }

                if(distance < min_distance) {
                    min_distance = distance;
                    min_center = j;
                }
            }

            for(l=0; l<n; l++)
                new_centers[min_center*n + l] += data_vectors[i*n + l];
            cluster_members[min_center][num_of_cluster_members[min_center]] = i;
            num_of_cluster_members[min_center]++;


        }

        /* Estimate the new centers */
        for(i=0; i<K; i++) {
            for(l=0; l<n; l++) {
                //GIANNIS
                if(num_of_cluster_members[i])
                    new_centers[i*n + l] /= (double)num_of_cluster_members[i];
                        //GIANNIS
                if(finite(new_centers[i*n + l]) == 0)
                    new_centers[i*n + l] = 0;
            }
        }

        //for(i=0; i<K; i++) {
        //	printf("Cluster members [%d]:  %d\n", i, num_of_cluster_members[i]);
        //}


        /* Here we print the total distance for each pass */
        for(i=0; i<K; i++) {
            for(j=0; j<num_of_cluster_members[i]; j++) {
                for(l=0; l<n; l++) {
                    total_distance += pow((data_vectors[cluster_members[i][j]*n + l] - centers[i*n + l]),2.0);
                }
            }
        }
        //printf("Total distance: %f\n", total_distance);
        total_distance=0;

        /* Check if converges */
        match=0;
        for(i=0; i<K; i++) {
            cur_match=0;
            for(j=0; j<n; j++) {
                if(new_centers[i*n + j] == centers[i*n + j])
                    cur_match++;
            }
            if(cur_match == n)
                match++;
        }

        /* If the centers remain the same: terminate */
        if(match == K)
            break;


        for(i=0; i<K; i++) {
            for(j=0; j<n; j++) {
                centers[i*n + j] = new_centers[i*n + j];
                new_centers[i*n + j]=0;
            }
            num_of_cluster_members[i]=0;
        }

        iterations++;
        if(iterations>2000) break;
    }                        /* telos tou while(1) */


    /* ----------------- YPOLOGISMOS VARIANCE -------------- */

    for(i=0; i<K; i++) {
        for(j=0; j<num_of_cluster_members[i]; j++) {
            for(l=0; l<n; l++) {
                variances[i*n + l] += pow((data_vectors[cluster_members[i][j]*n + l] - centers[i*n + l]),2.0);
            }
        }
    }

    for(i=0; i<K; i++) {
        for(j=0; j<n; j++) {
            //GIANNIS
            if(num_of_cluster_members[i]>1)
                variances[i*n + j] /= (double)num_of_cluster_members[i];
            else
            {
                variances[i*n+j]=0.01;
            }
        }
    }



    free(new_centers);
    delete[] random_centers;
    for(int i=0;i<K;i++) delete[] cluster_members[i];
    delete[] cluster_members;
}

double FunctionalRbf::nearestClass(double y)
{
    int ifound=-1;
    double dmin=1e+100;
    for(int i=0;i<(int)dclass.size();i++)
    {
        if(fabs(dclass[i]-y)<dmin)
        {
            dmin=fabs(dclass[i]-y);
            ifound=i;
        }
    }
    return dclass[ifound];
}

double  FunctionalRbf::output(Data &x)
{
        Data pattern = x;
        arma::vec neuronOuts(nodes);
        for(unsigned j = 0; j < nodes;j++){
            neuronOuts[j] = neuronOutput(weight,pattern,pattern.size(),j);
        }
        double tempOut = arma::dot(neuronOuts,Linear);
	return tempOut;
}
double FunctionalRbf::neuronOutput( vector<double> &x, vector<double> &patt, unsigned pattDim, unsigned offset ){
    double out = 0;
    for(unsigned i = 0; i < pattDim;i++){
        out += (patt[i] - x[offset*pattDim + i]) * (patt[i] - x[offset*pattDim + i]);
    }
    double df=(-out / (x[nodes*pattDim+offset] * x[nodes*pattDim+offset]) );
    return exp(df);
}

adept::adouble FunctionalRbf::aneuronOutput( vector<adept::adouble> &x, vector<double> &patt, unsigned pattDim, unsigned offset ){
    adept::adouble out = 0;
    for(unsigned i = 0; i < pattDim;i++){
        out += (patt[i] - x[offset*pattDim + i]) * (patt[i] - x[offset*pattDim + i]);
    }
    adept::adouble darg = out / (x[nodes*pattDim + offset] * x[nodes*pattDim + offset]);
    return exp(-out / (x[nodes*pattDim + offset] * x[nodes*pattDim + offset]) );
}

adept::adouble FunctionalRbf::afunmin( vector<adept::adouble> &x, vector<double> &x1 ){
    adept::adouble errorSum=0.0;

    bool ok;
    Linear = train(x1,ok);

    int icount = 0;
    for(unsigned i = 0; i < xpoint.size(); i++){
        Data pattern = xpoint[i];
        vector<adept::adouble> neuronOuts(nodes);
        for(unsigned j = 0; j < nodes;j++){
            neuronOuts[j] = aneuronOutput(x,pattern,pattern.size(),j);
        }
        adept::adouble tempOut = 0;
        for(int j = 0; j < nodes; j++) tempOut+= neuronOuts[j]*Linear[j];
        errorSum += ( tempOut - ypoint[i] ) * ( tempOut - ypoint[i] );
    }

    return errorSum;
}

arma::vec FunctionalRbf::train( vector<double> &x,bool &ok ){
    ok = true;
    arma::mat A = arma::zeros(xpoint.size(),nodes);
    arma::vec B(ypoint.size());
    for(unsigned i = 0; i < ypoint.size(); i++){
        B.at(i) = ypoint[i];
        for(unsigned j = 0; j < nodes;j++){
            A.at(i,j) = neuronOutput(x, xpoint[i], xpoint[0].size() , j);
        }
    }

    arma::vec RetVal;
    try{
        RetVal=arma::vec(arma::pinv(A)*B);

        // RetVal=arma::vec(arma::pinv(A,1e-10,"dc")*B);
    }
    catch(std::runtime_error & e)
    {
        RetVal = arma::zeros(arma::size(RetVal));
        ok = false;
    }
    if(RetVal.has_nan() || RetVal.has_inf()) {
        RetVal = arma::zeros(arma::size(RetVal));
    }
    return RetVal;
}

static double dmax(double a,double b)
{
    return a>b?a:b;
}

Data    FunctionalRbf::gradient(Data &x)
{
    Data g;
    g.resize(dimension);


    g = vector<double>(x.size(),0.0);
    adept::Stack s;
    std::vector<adept::adouble> ax(g.size());
    for(unsigned i = 0; i < x.size(); i++) ax[i] = x[i];
    s.new_recording();

    adept::adouble error = afunmin(ax, x);

    error.set_gradient(1.0);

    s.compute_adjoint();

    for(unsigned i = 0; i < x.size();i++) {
        g[i] = ax[i].get_gradient();
    }
    return g;
    for(int i=0;i<dimension;i++)
    {
        double eps=pow(1e-18,1.0/3.0)*dmax(1.0,fabs(x[i]));
        x[i]+=eps;
        double v1=funmin(x);
        x[i]-=2.0 *eps;
        double v2=funmin(x);
        g[i]=(v1-v2)/(2.0 * eps);
        x[i]+=eps;
    }
    return g;
}

FunctionalRbf::FunctionalRbf(Mapper *m)
    :Model (m)
{
    nodes = 10;
    rbf_factor = 3.0;
}


FunctionalRbf::~FunctionalRbf()
{
    if(xinput!=NULL)
    {
        delete[] xinput;
        delete[] yinput;
    }
}
