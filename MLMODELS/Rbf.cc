/* RBF Network Creation */

#include "Rbf.h"
#include "matrix_functions.h"

/* -------------------------------------------- */
/* Trains an RBF Neural Netowrk                 */
/* Input Parameters:                            */
/*      in_n      : number of inputs            */
/*                  (in the input layer)        */
/*      hid_n     : number of synapses          */ 
/*                  in the hidden layer         */
/*      out_n     : number of outputs           */ 
/*                  (in the output layer)       */
/*      samples_n : number of training samples  */
/*      centers   : pointer to centers matrix   */
/*                  (hid_n rows, in_n columns)  */ 
/*      variances : pointer to variances matrix */
/*                  (hid_n rows, in_n columns)  */ 
/*      weights   : pointer to weights matrix   */
/*                  (hid_n columns)             */
/*      input     : pointer to input vector     */
/*                  (in_n columns)              */
/*      output    : pointer to output vector    */
/*                  (out_n columns)             */
/* -------------------------------------------- */

double violate;
int train_rbf(int in_n, int hid_n, int out_n, int samples_n,
	      double * centers, double * variances, 
	      double * weights, double * input, double * output)
{

   int i,j;
   double result, var_diag=0.05;
   double in_cen[in_n], in_cen_tr[in_n];
   double G[samples_n][hid_n];
   double Gp[hid_n][samples_n];

   // -----------------------------------
	var_diag=0;
	for(i=0; i<hid_n; i++) {
		for(j=0; j<in_n; j++) {
			var_diag += variances[i*in_n + j];
		}
	}
	if(var_diag<0.00000001) var_diag=0.001;
   // -----------------------------------
 
	int violcount=0;
   for(i=0; i<samples_n; i++) {
	for(j=0; j<hid_n; j++) {
		/* Ypologizw ton pinaka Input-Centers gia kathe hidden neuron */
		matrix_sub(&input[i*in_n],&centers[j*in_n],&in_cen[0],1,in_n);

		/* Ypologizw ton anastrofo pinaka tou prohgoumenou */
		matrix_transpose(&in_cen[0],&in_cen_tr[0],in_n,1);

		/* Pollaplasiazw ta dyo parapanw gia na parw ena pinaka stoixeio */
		matrix_mult(&in_cen_tr[0],&in_cen[0],&result,1,in_n,1);


		G[i][j] = exp((-1.0*result)/(2.0 * var_diag));


	}
   }
   matrix_pseudo_inverse(&G[0][0],&Gp[0][0],samples_n,hid_n);
   
   /* Kanw ton pol/smo [Gp]*[Output] gia na brw ta weights */

   matrix_mult(&Gp[0][0],output,weights,hid_n,samples_n,out_n);

/* --------------------------------------------------------
   for(i=0; i<samples_n; i++) {
	for(j=0; j<hid_n; j++) {
		printf("%f\t", G[i][j]); 
	}
	printf("\n");
   }
   printf("\n\n");
   for(i=0; i<hid_n; i++) {
	for(j=0; j<samples_n; j++) {
		printf("%f\t", Gp[i][j]); 
	}
	printf("\n");
   }
   printf("\n\n");
   for(i=0; i<hid_n; i++) {
	for(j=0; j<out_n; j++) {
		printf("%f\t", weights[i*out_n + j]);
	}
	printf("\n");
   }
---------------------------------------------------------- */ 

 return 0;
}



/* -------------------------------------------- */
/* Creates an RBF Neural Netowrk                */
/* Input Parameters:                            */
/*      in_n      : number of inputs            */
/*                  (in the input layer)        */
/*      hid_n     : number of synapses          */ 
/*                  in the hidden layer         */
/*      out_n     : number of outputs           */ 
/*                  (in the output layer)       */
/*      centers   : pointer to centers matrix   */
/*                  (hid_n rows, in_n columns)  */ 
/*      variances : pointer to variances matrix */
/*                  (hid_n rows, in_n columns)  */ 
/*      weights   : pointer to weights matrix   */
/*                  (hid_n rows, out_n columns) */
/*      input     : pointer to input vector     */
/*                  (1 row, in_n columns)       */
/*      output    : pointer to output vector    */
/*                  (1 row, out_n columns)      */
/* -------------------------------------------- */

int create_rbf(int in_n, int hid_n, int out_n, 
	      double * centers, double * variances, 
	      double * weights, double * input, double * output)
{

   int i,j;
   double result, var_diag=0.05;
   double in_cen[in_n], in_cen_tr[in_n];
   double G[hid_n];

   // -----------------------------------
	var_diag=0;
	for(i=0; i<hid_n; i++) {
		for(j=0; j<in_n; j++) {
			var_diag += variances[i*in_n + j];
		}
	}
	if(var_diag<0.00000001) var_diag=0.001;
   // -----------------------------------

   for(i=0; i<hid_n; i++) {
	/* Ypologizw ton pinaka Input-Centers gia kathe hidden neuron */
	matrix_sub(&input[0],&centers[i*in_n],&in_cen[0],1,in_n);

	/* Ypologizw ton anastrofo pinaka tou prohgoumenou */
	matrix_transpose(&in_cen[0],&in_cen_tr[0],in_n,1);

	/* Pollaplasiazw ta dyo parapanw gia na parw ena pinaka stoixeio */
	matrix_mult(&in_cen_tr[0],&in_cen[0],&result,1,in_n,1);

	G[i] = exp((-1.0*result)/(2.0 * var_diag));

	/* Auto einai gia bias = 1 */
	/* G[hid_n-1]=1; */
   }
   
   /* Kanw ton pol/smo [G]*[Weights] gia na brw tin exodo */

   matrix_mult(&G[0],weights,output,1,hid_n,out_n);

 return 0;
}

/* K-Means Clustering Algorithm */

# include <vector>
using namespace std;

/* ---------------------------------------------- */
/* K-Means Clusteriing Algorithm                  */
/* Reads data from data_vectors matrix,           */
/* implements kmeans clustering and returns the   */
/* results to centers matrix along with the       */
/* variances into the variances matrix.           */
/* Input Parameters:                              */
/*      data_vectors : pointer to input data      */
/*                     matrix (m rows, n columns) */
/*      centers      : pointer to centers matrix  */
/*                     (K rows, n columns)        */
/*      variances    : pointer to the variances   */
/*                     matrix (K rows, n columns) */
/*      m            : number of data vectors     */
/*      n            : dimension of data vector   */
/*      K            : number of centers          */
/* ---------------------------------------------- */
/*
void myKmeans(double * data_vectors, double * centers,
            double * variances, int m, int n, int K)
{
    for(int i=0;i<K;i++)
        for(int j=0;j<n;j++)
            centers[i*n+j]=0.0;

    typedef vector<int> D;
    vector<D> member;
    member.resize(K);


    double *pattern =new double[n];
    double *old=new double[n];

    double sumdiff=1e+100;
    int iters=0;
    for(int i=0;i<m;i++)
    {
        int k=drand48()*K;
        if(k==K) k=K-1;
        member[k].push_back(i);
    }
    for(int i=0;i<K;i++)
    {
        for(int j=0;j<member[i].size();j++)
        {
            for(int k=0;k<n;k++)
            {
                pattern[k]=data_vectors[member[i][j]*n+k];
                centers[i*n+k]+=pattern[k];
            }
        }
        for(int j=0;j<n;j++) {
            centers[i*n+j]/=member[i].size();
        }
    }

    while(iters<20 && sumdiff>1000)
    {
        iters++;
        for(int i=0;i<K;i++)
            member[i].resize(0);

        for(int i=0;i<m;i++)
        {
            for(int j=0;j<n;j++) pattern[j]=data_vectors[i*n+j];
            double mindist=1e+100;
            int    imin=-1;
            for(int j=0;j<K;j++)
            {
                double s=0.0;
                for(int k=0;k<n;k++) s+=pow(centers[j*n+k]-pattern[k],2.0);
                if(s<mindist || j==0)
                {
                    mindist = s;
                    imin = j;
                }
            }
            member[imin].push_back(i);
        }

        sumdiff=0.0;
        for(int i=0;i<K;i++)
        {
            int count=0;
            for(int j=0;j<n;j++) {old[j]=centers[i*n+j];centers[i*n+j]=0.0;}
            double ilastclass=0.0;
            for(int j=0;j<member[i].size();j++)
            {
                count++;
                for(int k=0;k<n;k++)
                {
                    extern double *Output;
                    if(j==0) ilastclass=Output[member[i][j]];
                    else
                    if(Output[member[i][j]]!=ilastclass)
                    {
                        member[i][j]=-1;
                        count--;
                        break;
                    }
                    pattern[k]=data_vectors[member[i][j]*n+k];
                    centers[i*n+k]+=pattern[k];
                }
            }
            if(count)
            {
                for(int j=0;j<n;j++)
                {
                    centers[i*n+j]/=count;
                    sumdiff+=pow(old[j]-centers[i*n+j],2.0);
                }
            }
        }
    }

    for(int i=0;i<K;i++)
    {
        for(int j=0;j<n;j++) variances[i*n+j]=0.0;
        int count=0;
        for(int j=0;j<member[i].size();j++)
        {
            if(member[i][j]<0) continue;
            count++;
            for(int k=0;k<n;k++)
            {
                pattern[k]=data_vectors[member[i][j]*n+k];
                variances[i*n+k]+=pow(pattern[k]-centers[i*n+k],2.0);
            }
        }
        if(count)
        for(int k=0;k<n;k++) variances[i*n+k]/=count;
    }
    delete[] pattern;
    delete[] old;
}
*/

void Kmeans(double * data_vectors, double * centers,
            double * variances, int m, int n, int K)
{
    int i=0;
    int j=0;
    int l=0;
    int k=0;
    double * new_centers = (double*)malloc(sizeof(double)*K*n);
    int **cluster_members=new int*[K];
    for(int i=0;i<K;i++)
        cluster_members[i]=new int[m];
    int *num_of_cluster_members=new int[K];
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
    double var_diag=0;


    // Assign a random center to each example in the training set
    for(i=0; i<K; i++)
    {
        do{
            random_centers[i]=(int)((m-1) * drand48());
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
    //for(i=0; i<K; i++)  printf("Random center is: %d %d\n", i, random_centers[i]);


    // Create the initial random centers
    for(i=0; i<K; i++) {
        for(j=0; j<n; j++) {
            //if(random_centers[i]==m)
            //	printf("error \n");
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
            if(num_of_cluster_members[i])
                variances[i*n + j] /= (double)num_of_cluster_members[i];
            else
            {
                variances[i*n+j]=0;
            }
        }
    }



    free(new_centers);
    delete[] num_of_cluster_members;
    delete[] random_centers;
    for(int i=0;i<K;i++) delete[] cluster_members[i];
    delete[] cluster_members;

}



