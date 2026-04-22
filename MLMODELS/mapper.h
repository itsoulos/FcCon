# ifndef __MAPPER__H
# define __MAPPER__H
# include <GE/fparser.hh>
# include <CORE/problem.h>
# include <string>
using namespace std;
class Mapper
{
	private:
		int dimension;
		string vars;
		vector<FunctionParser*> parser;
		vector<int> foundX;
	public:
		Mapper(int d);
        void        setExpr(vector<string> s);
        int         map(Data &x,Data &x1);
		~Mapper();
};
# endif
