# ifndef __MAPPER__H
# define __MAPPER__H
# include <GE/fparser.hh>
# include <string>
using namespace std;
typedef vector<double> Matrix;
class Mapper
{
	private:
		int dimension;
		string vars;
		vector<FunctionParser*> parser;
		vector<int> foundX;
	public:
		Mapper(int d);
		void	setExpr(vector<string> s);
		int	map(Matrix x,Matrix &x1);
		~Mapper();
};
# endif
