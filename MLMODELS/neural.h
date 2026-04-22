# ifndef __NEURAL__H
# define __NEURAL__H
# include <MLMODELS/model.h>
class Neural :public Model
{
	public:
		Neural(Mapper *m);
		double	countViolate(double limit);
		virtual double train1();
		virtual double train2();
        virtual double output(Data &x);
        virtual void   getDeriv(Data &x,Data &g);
        void	 setWeights(Data x);
		~Neural();
};

# endif
