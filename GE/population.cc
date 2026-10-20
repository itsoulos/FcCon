# include <GE/population.h>
# include <stdlib.h>
# include <string.h>
# include <math.h>
# include <GRS/grs.h>
# include <GE/nnprogram.h>
# include <GE/integeranneal.h>
# include <GE/rule.h>
# include <GE/symbol.h>
# include <functional>
# include <algorithm>
# include <limits>
# include <utility>
# include <cmath>
# define MAX_RULE	256

mt19937 gen(random_device{}());

double randDouble(double a, double b) {
    uniform_real_distribution<> dist(a, b);
    return dist(gen);
}

int randInt(int a, int b) {
    uniform_int_distribution<> dist(a, b);
    return dist(gen);
}
vector<int> Population::neighbor(const vector<int>& x, int stepSize ) {
    vector<int> y = x;

    int i = randInt(0, x.size() - 1);

    // αλλαγή σε μία διάσταση
    y[i] += randInt(-stepSize, stepSize);
    if(y[i]<0) y[i]=0;

    return y;
}

vector<int> Population::simulatedAnnealing(
    vector<int> &x,
    double T0,
    double Tmin,
    double alpha,
    int iterPerTemp
    ) {
    double T = T0;

    vector<int> best = x;
    double bestVal = fitness(x);

    while (T > Tmin) {

        for (int k = 0; k < iterPerTemp; k++) {

            vector<int> y = neighbor(x);

            double fx = (fitness(x));
            double fy = (fitness(y));

            double delta = fabs(fy) - fabs(fx);

            if (delta < 0) {
                // καλύτερη λύση
                x = y;
            } else {
                // πιθανότητα αποδοχής
                double p = exp(-delta / T);

                if (randDouble(0,1) < p)
                    x = y;
            }

            // update best
            double val = fitness(x);
            if (fabs(val) < fabs(bestVal)) {
                best = x;
                bestVal = val;
            }
        }
        printf("T=%20.10lf BEST=%20.10lf\n",T,bestVal);
        // cooling
        T *= alpha;
    }

    return best;
}
/* Population constructor */
/* Input: genome count , genome size, pointer to Program instance */
Population::Population(int gcount,int gsize,Program *p)
{
	elitism=1;
	selection_rate = 0.1;
	mutation_rate  = 0.1;
	genome_count   = gcount;
	genome_size    = gsize;
	generation     = 0;
	program        = p;

	/* Create the population and based on genome count and size */
	/* Initialize the genomes to random */
    genome.resize(genome_count);
    children.resize(genome_count);
	for(int i=0;i<genome_count;i++)
	{
        genome[i].resize(genome_size);
        children[i].resize(genome_size);
			for(int j=0;j<genome_size;j++)
                genome[i][j]=rand()%MAX_RULE;
	}
    fitness_array.resize(genome_count);
}

/* Reinitialize the population to random */
void    	Population::reset()
{
	generation = 0;
	for(int i=0;i<genome_count;i++)
		for(int j=0;j<genome_size;j++)
				genome[i][j]=rand()%MAX_RULE;
	for(int i=0;i<genome_count;i++)
			fitness_array[i]=-1e+100;
}

/* Return the fitness of a genome */
double 	Population::fitness(vector<int> &g)
{
	return program->fitness(g);
}

/* The selection of the chromosomes according to the fitness value is performed */
void	 Population::select()
{
    vector<int> itemp;
	for(int i=0;i<genome_count;i++)
	{
		for(int j=0;j<genome_count-1;j++)
		{
			if(fitness_array[j+1]>fitness_array[j])
			{
				double dtemp;
				dtemp=fitness_array[j];
				fitness_array[j]=fitness_array[j+1];
				fitness_array[j+1]=dtemp;
                itemp = genome[j];
                genome[j]=genome[j+1];
                genome[j+1]=itemp;
			}
		}
	}
}

// Find codons used by DIGIT0/DIGIT1 expansions, without wrapping.
// This follows the deterministic grammar traversal used by Rule::printRule.
static vector<int> constantCodons(Program *program, const vector<int> &g) {
    vector<int> result;
    if(g.empty() || !program || !program->getStartSymbol()) return result;
    Symbol *start=program->getStartSymbol();
    if(start->getCountRules()<=0) return result;
    int pos=0;
    std::function<void(Rule*,int)> visit = [&](Rule *r,int depth) {
        if(!r || depth>1000 || pos >= (int)g.size()) return;
        for(int i=0;i<r->getLength();++i) {
            Symbol *sym=r->getSymbol(i);
            if(!sym || sym->getTerminalStatus()) continue;
            if(pos >= (int)g.size()) return; // no wrapping
            int n=sym->getCountRules();
            if(n<=0) return;
            int idx=pos++;
            if(sym->getName()=="DIGIT0" || sym->getName()=="DIGIT1")
                result.push_back(idx);
            visit(sym->getRule(g[idx]%n),depth+1);
        }
    };
    visit(start->getRule(g[0]%start->getCountRules()),0);
    return result;
}

/* Crossover operation: based on tournament selection */
/* Select the tournament_size based on the genome count : */
/*     (if genome_count > 100 ) tournament_size = 10   else   tournament_size = 4 */
/* Select 2 chromosomes based on the tournament size and cross them over based on the crossover probability */
/* There is 1 crossover point and it is random */
void        Population::crossover()
{
    if (crossMethod == "none") return;
        int parent[2];
        int nchildren=(int)((1.0 - selection_rate) * genome_count);
	if(!(nchildren%2==0)) nchildren++;
        const int tournament_size =(genome_count<=100)?4:20;
        int count_children=0;
        while(1)
        {
		// The two parents are selected here according to the tournament selection procedure
                for(int i=0;i<2;i++)
                {
                        double max_fitness=-1e+10;
                        int    max_index=-1;
			int r;
			// Select the best parents of  the candidates 
                        for(int j=0;j<tournament_size;j++)
                        {
				r=rand() % (genome_count);
                                if(j==0 || fitness_array[r]>max_fitness)
                                {
                                        max_index=r;
                                        max_fitness=fitness_array[r];
                                }
                        }
                        parent[i]=max_index;
			
                }
        int pt1;
        pt1=rand() % genome_size;

		// The one-point crossover is performed here (the point is pt1)
        std::copy(genome[parent[0]].begin(),
                  genome[parent[0]].begin() + pt1,
                  children[count_children].begin());
        std::copy(genome[parent[1]].begin() + pt1,
                  genome[parent[1]].end(),
                  children[count_children].begin() + pt1);
        std::copy(genome[parent[1]].begin(),
                  genome[parent[1]].begin() + pt1,
                  children[count_children + 1].begin());
        std::copy(genome[parent[0]].begin() + pt1,
                  genome[parent[0]].end(),
                  children[count_children + 1].begin() + pt1);
        if(crossMethod=="two_point" && genome_size>2) {
            int a=rand()%genome_size, b=rand()%genome_size;
            if(a>b) std::swap(a,b);
            for(int k=a;k<b;k++) {
                children[count_children][k]=genome[parent[1]][k];
                children[count_children+1][k]=genome[parent[0]][k];
            }
        } else if(crossMethod=="constants") {
            // Keep the first parent intact except for codons decoding as digits.
            // Rebuild children from their respective parents before exchanging digits.
            children[count_children]=genome[parent[0]];
            children[count_children+1]=genome[parent[1]];
            const auto p0=constantCodons(program,genome[parent[0]]);
            const auto p1=constantCodons(program,genome[parent[1]]);
            for(int k : p0) children[count_children][k]=genome[parent[1]][k];
            for(int k : p1) children[count_children+1][k]=genome[parent[0]][k];
        } else if(crossMethod=="uniform") {
            for(int k=0;k<genome_size;k++) if(rand()%2) {
                children[count_children][k]=genome[parent[1]][k];
                children[count_children+1][k]=genome[parent[0]][k];
            }
        }
		count_children+=2;
		if(count_children>=nchildren) break;
	}
	
	vector<int> g;
	g.resize(genome_size);
	
	for(int i=0;i<nchildren;i++)
	{
		//for(int j=0;j<g.size();j++) g[j]=children[i][j];
		//double f=fitness(g);
		//if(fabs(f)<fabs(fitness_array[genome_count-i-1]))

        genome[genome_count-i-1]=children[i];
	}
}

// Fitness-guided block recombination, adapted to FcCon's feature fitness.
// Unlike QGenClass, FcCon has no per-class semantic error interface.
void Population::targetedCrossItem(int pos, bool worst) {
    if(genome_size < 1 || genome_count < 2) return;
    int donor=0;
    if(worst) donor=(int)(std::min_element(fitness_array.begin(),fitness_array.end())-fitness_array.begin());
    else donor=(int)(std::max_element(fitness_array.begin(),fitness_array.end())-fitness_array.begin());
    if(donor==pos) donor=(donor+1)%genome_count;
    vector<int> best=genome[pos];
    double score=fitness(best);
    for(int t=0;t<operatorTrials;t++) {
        vector<int> trial=best;
        int a=rand()%genome_size;
        int len=1+rand()%std::max(1,std::min(genome_size-a,genome_size/5));
        std::copy(genome[donor].begin()+a,genome[donor].begin()+a+len,trial.begin()+a);
        double f=fitness(trial);
        if(std::isfinite(f) && fabs(f)<fabs(score)) {best=std::move(trial);score=f;}
    }
    genome[pos]=std::move(best);
    fitness_array[pos]=score;
}

void Population::targetedMutateItem(int pos, bool worst) {
    if(genome_size<1) return;
    vector<int> best=genome[pos];
    double score=fitness(best);
    for(int t=0;t<operatorTrials;t++) {
        vector<int> trial=best;
        int j=rand()%genome_size;
        if(worst) {
            int width=std::max(1,genome_size/10);
            for(int k=0;k<width;k++) trial[(j+k)%genome_size]=rand()%MAX_RULE;
        } else trial[j]=rand()%MAX_RULE;
        double f=fitness(trial);
        if(std::isfinite(f) && fabs(f)<fabs(score)) {best=std::move(trial);score=f;}
    }
    genome[pos]=std::move(best);
    fitness_array[pos]=score;
}

void        Population::setElitism(int s)
{
	elitism = s;
}

/* Mutate the current population */
/* Standard mutation algorithm: mutate all chromosomes in the population based on the mutation probability */
void    	Population::mutate()
{
    if (mutationMethod == "none") return;
	int start = elitism * (int)(genome_count*selection_rate);
	start = elitism;
	start = 1;
	for(int i=start;i<genome_count;i++)
	{
		for(int j=0;j<genome_size;j++)
		{
			double r=rand()*1.0/RAND_MAX;
			if(r<mutation_rate)
			{
                if(mutationMethod=="creep") {
                    int step=(rand()%2)?1:-1;
                    genome[i][j]=(genome[i][j]+step+MAX_RULE)%MAX_RULE;
                } else if(mutationMethod=="adaptive") {
                    int step=1+rand()%std::max(1,MAX_RULE/(generation+2));
                    genome[i][j]=(genome[i][j]+((rand()%2)?step:-step)+MAX_RULE)%MAX_RULE;
                } else genome[i][j]=rand()%MAX_RULE;
			}
		}
	}
}

/* Evaluate the fitness for all chromosomes in the current population */
void    	Population::calcFitnessArray()
{
	double dmin=1e+100;
	for(int i=0;i<genome_count;i++)
	{
        fitness_array[i]=fitness(genome[i]);

        if(fabs(fitness_array[i])<dmin)
		{
			dmin=fabs(fitness_array[i]);
		}
		
        if((i+1)%20==0)
		{
	//		printf(" %d:%.5lg ",i+1,dmin);
	//		fflush(stdout);
        }
		
	}
}

/* Return the current generation */
int	Population::getGeneration() const
{
	return generation;
}

/* Return the genome count */
int	Population::getCount() const
{
	return genome_count;
}

/* Return the size of the population */
int	Population::getSize() const
{
	return genome_size;
}

/* Evolve the next generation */
void        Population::nextGeneration()
{
    calcFitnessArray();
    select();
    crossover();
    if(generation) mutate();
    ++generation;
    // Optional periodic crossItem: choose distinct random population positions.
    // Disabled if either parameter is zero. Does not change local search.
    if (crossItems > 0 && crossInterval > 0 && generation % crossInterval == 0)
    {
        const int count = crossItems < genome_count ? crossItems : genome_count;
        vector<int> positions(genome_count);
        for (int i=0; i<genome_count; ++i) positions[i]=i;
        for (int i=0; i<count; ++i)
        {
            const int j = i + rand() % (genome_count-i);
            std::swap(positions[i],positions[j]);
            crossItem(i==0?0:positions[i]);
        }
	select();
    }
    if(generation%50==0)
    {
	    for(int i=0;i<5;i++)
		    localSearch(i==0?0:rand()%genome_count);
	    select();
    }
}

void    	Population::replaceWorst()
{
	vector<int> xtrial;
	xtrial.resize(genome_size);
	int randpos;
	randpos=rand() % genome_count;
	for(int i=0;i<genome_size;i++)
	{
		double gamma;
		gamma=-0.5+2.0*drand48();
		xtrial[i]=(int)(fabs((1.0+gamma)*genome[0][i]-gamma*genome[randpos][i]));
	}
	double ftrial = fitness(xtrial);
	if(ftrial>fitness_array[genome_count-1])
	{
		for(int i=0;i<genome_size;i++)
		genome[genome_count-1][i]=xtrial[i];
		fitness_array[genome_count-1]=ftrial;
	}
}

/* Set the mutation rate */
void        Population::setMutationRate(double r)
{
	if(r>=0 && r<=1) mutation_rate = r;
}

/* Set the selection rate */
void    	Population::setSelectionRate(double r)
{
	if(r>=0 && r<=1) selection_rate = r;
}

/* Return the selection rate */
double	Population::getSelectionRate() const
{
	return selection_rate;
}

/* Return the mutation rate */
double	Population::getMutationRate() const
{
	return mutation_rate;
}

/* Return the best fitness for this population */
double	Population::getBestFitness() const
{
	return  fitness_array[0];
}

/* Return the best chromosome */
vector<int> Population::getBestGenome() const
{
	vector<int> g;g.resize(genome_size);
	for(int i=0;i<genome_size;i++) g[i]=genome[0][i];
	return g;
}

class PopulationProblem: public Problem
{
	private:
		Population *pop;
		vector<int> currentGenome;
	public:	
	PopulationProblem(Population *p,vector<int> &x)
	{
		pop = p;
		currentGenome = x;
		setDimension(x.size());
		lmargin.resize(x.size());
		rmargin.resize(x.size());
		for(int i=0;i<x.size();i++)
		{
			lmargin[i]=0;
			rmargin[i]=MAX_RULE;
		}
	}

 double dmax(double a,double b){return a>b?a:b;}
    virtual double funmin(Data &x)
	{
		for(int i=0;i<x.size();i++) {
			currentGenome[i]=(int)x[i];
			if(currentGenome[i]<0) currentGenome[i]=0;
		}
		double f= -pop->fitness(currentGenome);
		return f;
		
	}
    virtual void        granal(Data &x,Data &g)
	{
		for(int i=0;i<x.size();i++)
		{
		double eps=pow(1e-18,1.0/3.0)*dmax(1.0,fabs(x[i]));
		x[i]+=eps;
		double v1=funmin(x);
		x[i]-=2.0 *eps;
		double v2=funmin(x);
		g[i]=(v1-v2)/(2.0 * eps);
		x[i]+=eps;
		}
	}

};



vector<int> Population::discreteGradient(vector<int>& x)
{
    int n = x.size();
    vector<int> grad(n);

    double fx = fitness(x);

    for (int i = 0; i < n; i++) {

        vector<int> x_plus = x;
        vector<int> x_minus = x;

        x_plus[i] += 1;
        x_minus[i] -= 1;
        if(x_minus[i]<0) x_minus[i]=0;

        double f_plus = fitness(x_plus);
        double f_minus = fitness(x_minus);

        if (fabs(f_plus) < fabs(fx))
            grad[i] = +1;
        else if (fabs(f_minus) < fabs(fx))
            grad[i] = -1;
        else
            grad[i] = 0;
    }
    return grad;
}

vector<int> Population::discreteStep(vector<int>& x,vector<int>& grad)
{
    vector<int> res = x;

    for (int i = 0; i < (int)x.size(); i++)
    {
        res[i] += grad[i];
    }

    return res;
}

void Population::integerLocalSearch(vector<int> &x,int maxSteps)
{
    double bestVal = fitness(x);
    int stepSize = 100;
    for (int step = 0; step < maxSteps; step++) {

        vector<int> grad = discreteGradient(x);

        vector<int> candidate = x;

        for (int i = 0; i < x.size(); i++)
        {
            candidate[i] += grad[i] * stepSize;
            if(candidate[i]<0) candidate[i]=0;
        }

        double val = fitness(candidate);

        if (fabs(val) < fabs(bestVal)) {
            x = candidate;
            bestVal = val;
        } else {
            stepSize = max(1, stepSize / 2); // learning rate decay
        }
        printf("GD[%4d]=%20.10lg\n",step,bestVal);
    }


}


vector<int> Population::integerAdam(
    vector<int> x,
    int steps,
    double alpha,
    double beta1,
    double beta2,
    double eps
    ) {
    int n = x.size();

    vector<double> m(n, 0.0);
    vector<double> v(n, 0.0);

    double bestVal = fitness(x);

    for (int t = 1; t <= steps; t++) {

        vector<int> g_int = discreteGradient(x);
printf("ADAM[%d] TRY: %20.10lf\n",t,bestVal);
        // convert gradient to double
        vector<double> g(n);
        for (int i = 0; i < n; i++)
            g[i] = (double)g_int[i];

        // update moments
        for (int i = 0; i < n; i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * g[i];
            v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
        }

        // bias correction
        vector<double> m_hat(n), v_hat(n);
        for (int i = 0; i < n; i++) {
            m_hat[i] = m[i] / (1 - pow(beta1, t));
            v_hat[i] = v[i] / (1 - pow(beta2, t));
        }

        // candidate update
        vector<int> candidate = x;

        for (int i = 0; i < n; i++) {

            double step = alpha * m_hat[i] / (sqrt(v_hat[i]) + eps);
            double p = fabs(step);

            if (rand()*1.0/RAND_MAX < p) {
                candidate[i] += (step > 0 ? 1 : -1);
            }

            if(candidate[i]<0) candidate[i]=0;
        }

        double val = fitness(candidate);

        if (fabs(val) < fabs(bestVal)) {
            x = candidate;
            bestVal = val;
        }
        else {
            alpha *= 0.7; // decay learning rate
        }
    }

    return x;
}

// Adam-inspired local search for integer GE codons.
// Finite differences estimate beneficial integer directions, while
// Adam moments prioritize stable directions. Only a few codons change
// per accepted move, preserving most of the grammatical structure.
vector<int> Population::discreteAdam(vector<int> x, int steps)
{
    const int n = static_cast<int>(x.size());
    if (n == 0 || steps <= 0) return x;

    const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    vector<double> first(n, 0.0), second(n, 0.0);
    double best = fitness(x);
    if (!std::isfinite(best)) return x;

    const int maxCoordinates = std::min(n, 16);
    const int maxChanges = std::min(n, 3);
    for (int t = 1; t <= steps; ++t) {
        vector<int> coords(n);
        for (int i = 0; i < n; ++i) coords[i] = i;
        // Partial Fisher-Yates shuffle; avoid evaluating all codons.
        for (int i = 0; i < maxCoordinates; ++i) {
            const int j = i + rand() % (n - i);
            std::swap(coords[i], coords[j]);
        }

        vector<double> gradient(n, 0.0);
        for (int k = 0; k < maxCoordinates; ++k) {
            const int i = coords[k];
            vector<int> candidate = x;
            double plusScore = std::numeric_limits<double>::infinity();
            if (x[i] < std::numeric_limits<int>::max()) {
                candidate[i] = x[i] + 1;
                const double fp = fitness(candidate);
                if (std::isfinite(fp) && fabs(fp) < fabs(best)) {
                    gradient[i] = 1.0;
                    plusScore = fabs(fp);
                }
            }
            if (x[i] > 0) {
                candidate[i] = x[i] - 1;
                const double fm = fitness(candidate);
                if (std::isfinite(fm) && fabs(fm) < fabs(best) &&
                    (gradient[i] == 0.0 || fabs(fm) < plusScore)) {
                    gradient[i] = -1.0;
                }
            }
        }

        vector<std::pair<double, int>> ranked;
        for (int i = 0; i < n; ++i) {
            first[i] = beta1 * first[i] + (1.0 - beta1) * gradient[i];
            second[i] = beta2 * second[i] + (1.0 - beta2) * gradient[i] * gradient[i];
            if (gradient[i] == 0.0) continue;
            const double mhat = first[i] / (1.0 - pow(beta1, t));
            const double vhat = second[i] / (1.0 - pow(beta2, t));
            const double priority = mhat / (sqrt(vhat) + eps);
            if (priority * gradient[i] > 0.0)
                ranked.push_back(std::make_pair(fabs(priority), i));
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const std::pair<double,int>& a, const std::pair<double,int>& b) {
                      return a.first > b.first;
                  });

        bool improved = false;
        const int count = std::min(maxChanges, static_cast<int>(ranked.size()));
        // Try single-codon changes first to minimize disruption.
        for (int k = 0; k < count; ++k) {
            const int i = ranked[k].second;
            const int direction = gradient[i] > 0 ? 1 : -1;
            for (int step : {1, 2, 4}) {
                const long long next = static_cast<long long>(x[i]) + direction * step;
                if (next < 0 || next > std::numeric_limits<int>::max()) continue;
                vector<int> candidate = x;
                candidate[i] = static_cast<int>(next);
                const double f = fitness(candidate);
                if (std::isfinite(f) && fabs(f) < fabs(best)) {
                    x.swap(candidate);
                    best = f;
                    improved = true;
                    break;
                }
            }
            if (improved) break;
        }
        if (!improved && ranked.empty()) break;
    }
    return x;
}

void    Population::setLocalMethod(string s)
{
    localMethod = s;
}

void        Population::crossItem(int pos)
{
    if (pos < 0 || pos >= genome_count || genome_size < 1 || genome_count < 2)
        return;
    if (crossMethod == "none")
        return;

    // These methods are selected HERE, not as extra operations
    // on every offspring in nextGeneration().
    if (crossMethod == "targeted" || crossMethod == "targetedWorst") {
        targetedCrossItem(pos, crossMethod == "targetedWorst");
        return;
    }

    printf("CROSS[%d] (%s) = ", pos, crossMethod.c_str());
    fflush(stdout);

    vector<int> trial(genome_size);
    const int attempts = 100; // preserve original crossItem trial count
    for (int iter = 0; iter < attempts; ++iter) {
        const int donor = rand() % genome_count;
        if (donor == pos) continue;

        const vector<int> source = genome[pos];
        const vector<int> other = genome[donor];

        for (int orientation = 0; orientation < 2; ++orientation) {
            const vector<int> &a = orientation == 0 ? source : other;
            const vector<int> &b = orientation == 0 ? other : source;
            trial = a;

            if (crossMethod == "standard") {
                const int cut = rand() % genome_size;
                for (int j = cut; j < genome_size; ++j) trial[j] = b[j];
            } else if (crossMethod == "two_point") {
                int left = rand() % genome_size;
                int right = rand() % genome_size;
                if (left > right) std::swap(left, right);
                for (int j = left; j <= right; ++j) trial[j] = b[j];
            } else if (crossMethod == "uniform") {
                for (int j = 0; j < genome_size; ++j)
                    if (rand() % 2) trial[j] = b[j];
            } else if (crossMethod == "constants") {
                const vector<int> positions = constantCodons(program, a);
                if (positions.empty()) break;
                for (int j : positions)
                    if (j >= 0 && j < genome_size) trial[j] = b[j];
            } else {
                return; // unknown crossover method
            }

            const double f = fitness(trial);
            if (std::isfinite(f) && fabs(f) < 1e+10 &&
                fabs(f) < fabs(fitness_array[pos])) {
                genome[pos] = trial;
                fitness_array[pos] = f;
                printf("%lf ", f);
                fflush(stdout);
                break;
            }
        }
    }
    printf("\n");
}

// The consumed GE codons (no wrapping), including the initial start rule.
// If the grammar traversal cannot identify any, use all codons as fallback.
static vector<int> activeCodons(Program *program, const vector<int> &g) {
    vector<int> positions;
    if(g.empty() || !program || !program->getStartSymbol()) return positions;
    Symbol *start=program->getStartSymbol();
    if(start->getCountRules()<=0) return positions;
    positions.push_back(0);
    int cursor=1;
    std::function<void(Rule*,int)> visit = [&](Rule *r,int depth) {
        if(!r || depth>1000 || cursor >= (int)g.size()) return;
        for(int i=0;i<r->getLength();++i) {
            Symbol *sym=r->getSymbol(i);
            if(!sym || sym->getTerminalStatus()) continue;
            int n=sym->getCountRules();
            if(n<=0 || cursor >= (int)g.size()) return;
            const int idx=cursor++;
            positions.push_back(idx);
            visit(sym->getRule(g[idx]%n),depth+1);
        }
    };
    visit(start->getRule(g[0]%start->getCountRules()),0);
    return positions;
}

// Fitness is maximized in the original mutateItem implementation.
// Use a fixed evaluation budget for a meaningful comparison of methods.
void Population::mutateItemVariant(int pos, const string &method) {
    if(pos<0 || pos>=genome_count || genome_size<=0) return;
    vector<int> best=genome[pos];
    double bestFitness=fitness_array[pos];
    if(!std::isfinite(bestFitness)) return;
    const int budget=1000;
    const int k=10;
    int evaluations=0, improvements=0, stagnation=0;
    int adaptiveRadius=4;
    printf("LOCAL_%s[%d] = ",method.c_str(),pos);
    fflush(stdout);
    while(evaluations < budget && stagnation < 5*genome_size) {
        vector<int> candidates;
        if(method=="mutate_active") candidates=activeCodons(program,best);
        if(candidates.empty()) {
            candidates.resize(genome_size);
            for(int j=0;j<genome_size;++j) candidates[j]=j;
        }
        int i=candidates[rand()%candidates.size()];
        const int original=best[i];
        int winningCodon=original;
        double winningFitness=bestFitness;
        const int attempts=std::min(k,budget-evaluations);
        for(int t=0;t<attempts;++t) {
            int value=original;
            if(method=="mutate_adaptive") {
                int step=1+rand()%adaptiveRadius;
                value=(original+((rand()%2)?step:-step)+MAX_RULE)%MAX_RULE;
                if(t==attempts-1) value=rand()%MAX_RULE;
            } else if(method=="mutate_active") {
                // Prefer local steps, but occasionally resample the codon.
                if(t%4==0) value=rand()%MAX_RULE;
                else value=(original+((rand()%2)?1:-1)*(1+rand()%8)+MAX_RULE)%MAX_RULE;
            } else { // mutate_best: best of K uniform replacements
                value=rand()%MAX_RULE;
            }
            if(value==original) continue;
            vector<int> trial=best;
            trial[i]=value;
            double f=fitness(trial);
            ++evaluations;
            if(std::isfinite(f) && f>winningFitness) {
                winningFitness=f;
                winningCodon=value;
            }
        }
        if(winningCodon!=original) {
            best[i]=winningCodon;
            bestFitness=winningFitness;
            ++improvements;
            stagnation=0;
            if(method=="mutate_adaptive") adaptiveRadius=std::min(128,adaptiveRadius*2);
        } else {
            ++stagnation;
            if(method=="mutate_adaptive") adaptiveRadius=std::max(1,adaptiveRadius/2);
        }
    }
    genome[pos]=best;
    fitness_array[pos]=bestFitness;
    printf("%lf (evaluations=%d improvements=%d)\n",bestFitness,evaluations,improvements);
}

void    	Population::mutateItem(int pos)
{
	vector<int> g;
	g.resize(genome_size);
	for(int i=0;i<genome_size;i++) g[i]=genome[pos][i];
    printf("LOCAL[%d] = ",pos);
    fflush(stdout);
    for(int j=0;j<10;j++)
       for(int i=0;i<genome_size;i++)
       {
                {
                        int ik=0;
                        double f;
                        do
                        {
                                g[i]=rand() % MAX_RULE;
                                ik++;
                                if(ik==10) break;
                                f=fitness(g);
                        }while(f<=fitness_array[pos]);
                        if(ik!=10)
                        {
                                fitness_array[pos]=f;
				printf(" %lf ",f);
				
				fflush(stdout);
                                genome[pos][i]=g[i];
                        }
                        else g[i]=genome[pos][i];
                }
        }
	printf("\n");
}

void        Population::localSearch(int pos)
{
	vector<int> g;
	g.resize(genome_size);
	for(int i=0;i<genome_size;i++) g[i]=genome[pos][i];

    if(localMethod == "crossover")
    {
	    crossItem(pos);
    }
    else
    if(localMethod == "mutate")
    {
	    mutateItem(pos);
    }
    else
    if(localMethod == "mutate_best" || localMethod == "mutate_adaptive" || localMethod == "mutate_active")
    {
        mutateItemVariant(pos,localMethod);
    }
    else
    if(localMethod == "siman")
    {
            double f = fitness_array[pos];
            IntegerAnneal lt(program);
            lt.setPoint(g,fitness_array[pos]);
            lt.Solve();
            lt.getPoint(g,fitness_array[pos]);
            //g=simulatedAnnealing(g);
         	g=integerAdam(g);
            fitness_array[pos]=fitness(g);
            genome[pos]=g;

            printf("SIMAN[%d] %lf=>%lf\n",pos,f,fitness_array[pos]);
    }
    else
    if(localMethod=="de")
    {
        int randomA,randomB,randomC;
         do
         {
             randomA =  rand() % genome_count;
             randomB =  rand() % genome_count;
             randomC =  rand() % genome_count;
         }while(randomA == randomB || randomB == randomC || randomC == randomA);
         double CR= 0.9;
         double F = 0.8;
         int randomIndex = rand() % genome_size;
     for(int i=0;i<genome_size;i++)
     {
         if(i==randomIndex || rand()*1.0/RAND_MAX <=CR)
         {
             int old_value = genome[pos][i];
             F = -0.5 + 2.0 * rand()*1.0/RAND_MAX;
             genome[pos][i]=genome[randomA][i]+abs(F*(genome[randomB][i]-genome[randomC][i]));
             if(genome[pos][i]<0)
             {
              genome[pos][i]=old_value;
              continue;
             }

             for(int j=0;j<genome_size;j++) g[j]=genome[pos][j];
             double trial_fitness=fitness(g);
             if(fabs(trial_fitness)<fabs(fitness_array[pos]))
             {
                printf("DE. NEW VALUE[%d] = %lf=>%lf\n",pos,fitness_array[pos],trial_fitness);
                fitness_array[pos]=trial_fitness;
             }
             else	genome[pos][i]=old_value;
         }
    }
    }
    else
        if(localMethod=="gd")
    {
        integerLocalSearch(g);
        double ff = fitness(g);
        if(fabs(ff)<fabs(fitness_array[pos]))
        {
            printf("GD. NEW VALUE[%d] = %lf=>%lf\n",pos,fitness_array[pos],ff);
            fitness_array[pos]=ff;
            for(int j=0;j<g.size();j++) genome[pos][j]=g[j];
        }
    }
    else if (localMethod == "discrete_adam")
    {
        g = discreteAdam(g);
        const double ff = fitness(g);
        if (std::isfinite(ff) && fabs(ff) < fabs(fitness_array[pos])) {
            printf("DISCRETE_ADAM. NEW VALUE[%d] = %lf=>%lf\n", pos, fitness_array[pos], ff);
            fitness_array[pos] = ff;
            genome[pos] = g;
        }
    }
    else
        if(localMethod == "adam")
    {
        g=integerAdam(g);
        double ff = fitness(g);
        if(fabs(ff)<fabs(fitness_array[pos]))
        {
            printf("ADAM. NEW VALUE[%d] = %lf=>%lf\n",pos,fitness_array[pos],ff);
            fitness_array[pos]=ff;
            for(int j=0;j<g.size();j++) genome[pos][j]=g[j];
        }
    }
	
}

/* Evaluate and return the best fitness for all chromosomes in the population */
double	Population::evaluateBestFitness()
{
    return fitness(genome[0]);
}

/* Destructor */
Population::~Population()
{

}
