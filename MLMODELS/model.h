# ifndef __MODEL__H
# define __MODEL__H

# include <CORE/problem.h>
# include <MLMODELS/mapper.h>
# include <CORE/dataset.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <map>
extern Data		xmax;
extern Data		xmin;
extern Data		xmean;
extern Data		xstd;
extern Data		xcurrent;


class Model :public Problem
{
	protected:
        int             isvalidation;
        Data            weight;
        int             num_weights;
        int             pattern_dimension;
        int             original_dimension;

        vector<Data> 	xpoint;
        Data          ypoint;
        Dataset     *trainSet;
        Dataset     *testSet;
        vector<Data> xall;
	public:

		Mapper	*mapper;
		Model(Mapper *m);
        void        setPatternDimension(int d);
        void        setNumOfWeights(int w);
        void    setTrainSet(Dataset *t);
        void    setTestSet(Dataset *t);
        int     getPatternDimension() const;
        int     getOriginalDimension() const;
        int     getNumOfWeights() const;
        int     getNumPatterns() const;
        Data    	getWeights();
        Data    	getXpoint(int pos);
		double  getYPoint(int pos);
		double  getModelAtPoint(int pos);
		/*	BASIKH SHMEIOSI
		 *	train1: Kaleitai gia tin ekpaideysi toy genetikou.
		 *	train2: Kaleitai otan teleiosei i parapano ekpaideysi.
		 * */
		virtual	double 	train1()=0;
		virtual double	train2()=0;
        virtual double	output(Data &x)=0;
        virtual void    getDeriv(Data  &x,Data &g)=0;
		
        virtual double	funmin(Data &x);
        virtual void    granal(Data &x,Data &g);
		double  valError();
        void    enableValidation();
        double	testError();
        double	classTestError();
        void    	randomizeWeights();
        void    	printConfusionMatrix(vector<double> &dclass,
                                        vector<double> &T,vector<double> &O,
                                         vector<double> &precision,
                                         vector<double> &recall);
        void    getPrecisionRecall(Dataset *t,
                    double &avg_precision,double &avg_recall,
                    double &avg_fscore);

        double distance(const std::vector<double>& a, const std::vector<double>& b);

        double  getAverageClassError(Data &x);
        bool    mapTrainSet();
		~Model();
};

# endif
