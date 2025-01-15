# ifndef __NNPROGRAM__H
# include <GE/program.h>
# include <GE/cprogram.h>
# include <MLMODELS/model.h>
# include <MLMODELS/neural.h>
# include <MLMODELS/rbf_model.h>
# include <MLMODELS/knn.h>
# include <MLMODELS/mapper.h>
# include <vector>
using namespace std;

# define MODEL_NEURAL		1
# define MODEL_RBF		2
# define MODEL_KNN		3

class NNprogram	:public Program
{
	private:
		vector<string> pstring;
		vector<int>    pgenome;
		int	model_type;
		int	pattern_dimension;
		Cprogram *program;
		Model	 *model;
		Mapper	 *mapper;
	public:
		NNprogram(int type,int pdimension,char *filename);
		string	printF(vector<int> &genome);
		virtual double 	fitness(vector<int> &genome);
		Model	*getModel();
		Mapper	*getMapper();
        virtual ~NNprogram();
};
# define __NNPROGRAM__H
# endif
