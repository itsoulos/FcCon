# ifndef __FC_RBF__H
# define __FC_RBF__H
# include <MLMODELS/model.h>
# include <MLMODELS/Rbf.h>
class Rbf :public Model
{
	private:
	        double *input,*centers,*variances,*weights;
	public:
		Rbf(Mapper *m);
		double	setWeightValuesFromPattern(double *pattern,int size);
		virtual double train1();
		virtual double train2();
        virtual double output(Data &x);
        virtual void   getDeriv(Data &x,Data &g);
		~Rbf();
};

# endif
