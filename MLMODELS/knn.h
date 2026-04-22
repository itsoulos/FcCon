# ifndef __KNN__H
# define __KNN__H
# include <MLMODELS/model.h>
typedef vector<int> IntVector;
class KNN :public Model
{
	private:
        void        sortArray(Data &x,vector<int> &index);
	public:
		KNN(Mapper *m);
        void        makeDistance(vector<Data> &testx,vector<Data> &distance);
        void        loadTest(char *filename,vector<Data> &testx,Data &testy);
		virtual double train1();
		virtual double train2();
        virtual double output(Data &x);
        virtual void   getDeriv(Data &x,Data &g);
        double	KNNtestError(vector<Data> &testx,Data &testy,vector<Data> &distance);
		~KNN();
};

# endif
