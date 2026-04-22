# ifndef __PROBLEM__H
# include <stdio.h>
# include <string.h>
# include <stdlib.h>
# include <math.h>
# include <vector>
using namespace std;
typedef vector<double> Data;
class Problem;
typedef double (Problem::*FUNCTION)(Data &);
typedef void   (Problem::*GRADIENT)(Data &,Data &);


# define MARGINS_NONE	0
# define MARGINS_LEFT	1
# define MARGINS_RIGHT	2

class Problem
{
	protected:
		int      dimension;
        int     has_margins;
        Data 	 lmargin;
        Data 	 rmargin;
	public:
		Problem(int d);
		Problem();
        void    	setDimension(int d);
        void    	setMargin(Data x1,Data x2);
        void    	setLeftMargin(Data x1);
        void    	setRightMargin(Data x2);
        virtual double  funmin(Data &);
        virtual void   granal(Data & x,Data &g);
		
		int		getDimension() const;
        void		getLeftMargin(Data &x1);
        void		getRightMargin(Data &x2);
        FUNCTION    	getFunmin();
        GRADIENT    	getGranal();
        void            getSample(Data &x);
        void            getSample(double *x);
        void            getSample(Data &x1,double rc,Data &x);
        int         isPointIn(Data &x);
        int         hasMargins();
        double		getLsStep(Data x,Data p);
        void            bound(Data &x);
        double		getGRMS(Data x,Data g);
		void		plot();
		void		plot(char *filename);
		double		getVolume();
		double		getLogVolume();
        double		getGradientCriterion(Data x1,Data x2);
		~Problem();
		
};

typedef struct MinInfo
{
	Problem *p;
	int	fevals;
	int	gevals;
	int	iters;
};

extern "C"{
extern double tolmin(Data &,MinInfo&);
extern double small_tolmin(Data &,MinInfo&);
}
# define __PROBLEM__H
# endif
