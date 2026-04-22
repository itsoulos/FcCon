# ifndef __MODEL__H
# define __MODEL__H

# include <CORE/problem.h>
# include <MLMODELS/mapper.h>
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

struct Sample {
    std::vector<double> features;
    double label;
};
class Model :public Problem
{
	protected:
        int             isvalidation;
        Data          weight;
        int             num_weights;
        int             pattern_dimension;
        int             original_dimension;
        vector<Data> 	origx;
        Data          origy;
        vector<Data> 	xpoint;
        Data          ypoint;
        Data          dclass;
	public:

		Mapper	*mapper;
		Model(Mapper *m);
        void        setPatternDimension(int d);
        void        setNumOfWeights(int w);
		void 	readPatterns(char *filename);
        void        replacePattern(int pos,Data x,double y);
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
        virtual void        getDeriv(Data  &x,Data &g)=0;
		
        virtual double	funmin(Data &x);
        virtual void    granal(Data &x,Data &g);
        void        transform(Data x,Data &xx);
		double  valError();
        void        enableValidation();
		double	testError(char *filename);
		double	classTestError(char *filename,double &precision,double &recall);
        void    	print(char *train,char *itest,char *otest);
        void    	randomizeWeights();
        void    	printConfusionMatrix(vector<double> &dclass,
                                        vector<double> &T,vector<double> &O,
                                         vector<double> &precision,
                                         vector<double> &recall);
        void    getPrecisionRecall(const char *filename,
                    double &avg_precision,double &avg_recall,
                    double &avg_fscore);
        void    getPrecisionRecall(
                                double &avg_precision,double &avg_recall,
                                double &avg_fscore);
        double distance(const std::vector<double>& a, const std::vector<double>& b);
        std::vector<int> kNearest(
            const std::vector<Sample>& minority,
            int index,
            int k);
        std::vector<Sample> applySMOTE(
            const std::vector<Sample>& data,
            int k = 5
            );
        void    enableSmote();
        double  getAverageClassError(Data &x);
		~Model();
};

# endif
