# ifndef __POPULATION__H
# include <GE/program.h>

/* The Population class holds the current population. */
/* The mutation, selection and crossover operators are defined here */
class Population
{
	private:
		int	**children;
		int	**trialx;
		int	**genome;
		int	*valid;
		double *fitness_array;
		double	mutation_rate,selection_rate;
		int	genome_count;
		int	genome_size;
		int	generation;
		Program	*program;

        void    	select();
        void    	crossover();
        void    	mutate();
        void    	calcFitnessArray();
        void    	replaceWorst();
		int	elitism;
        string localMethod="none";
        void    	localSearch(int gpos);
	public:
		Population(int gcount,int gsize,Program *p);
		double 	fitness(vector<int> &g);
		void	setElitism(int s);
        void    setLocalMethod(string s);
        int     getGeneration() const;
        int     getCount() const;
        int     getSize() const;
        void    nextGeneration();
        void    	setMutationRate(double r);
        void    	setSelectionRate(double r);
		double	getSelectionRate() const;
		double	getMutationRate() const;
		double	getBestFitness() const;
		double	evaluateBestFitness();
		vector<int> getBestGenome() const;
        void	 reset();
        vector<int> discreteGradient( vector<int>& x);
        vector<int> discreteStep(vector<int>& x,vector<int>& grad);
        void integerLocalSearch(vector<int> &x,int maxSteps = 20);
        vector<int> integerAdam(
            vector<int> x,
            int steps = 20,
            double alpha = 0.5,
            double beta1 = 0.9,
            double beta2 = 0.999,
            double eps = 1e-8
            ) ;
		~Population();
		
};
# define __POPULATION__H
# endif
