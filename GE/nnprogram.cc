#include <algorithm>
#include <cmath>
#include <limits>
# include <GE/nnprogram.h>
# include <RbfNetwork/rbf.h>
NNprogram::NNprogram(int type,int pdimension,
                     Dataset *train,Dataset *test)
{
    ruleTrainSet = train;
    ruleWeight = 0.0;
    neuralRulesMode = false;
    neuralRulesTolerance = 0.01;
    int d = train->dimension();
	mapper=new Mapper(d);
    model_type = type;
	if(type == MODEL_NEURAL)
		model = new Neural(mapper);
	else
	if(type == MODEL_RBF)
		model = new Rbf(mapper);
    else
        if(type==MODEL_AIRBF)
        model=new AiRbf(mapper);
    else
        if(type==MODEL_FRBF)
        model = new FunctionalRbf(mapper);
    else
        if(type==MODEL_MRBF)
        model = new RbfNetwork(mapper);
	else
		model = new KNN(mapper);
	pattern_dimension = pdimension;
	program = new Cprogram(d,pdimension);
	setStartSymbol(program->getStartSymbol());
    model->setTrainSet(train);
    model->setTestSet(test);
    model->setPatternDimension(pdimension);
	pstring.resize(pattern_dimension);
	pgenome.resize(0);
}

string	NNprogram::printF(vector<int> &genome)
{
	string ret="";
	if(pgenome.size()!=genome.size()/pattern_dimension)
		pgenome.resize(genome.size()/pattern_dimension);
	for(int i=0;i<pattern_dimension;i++)
	{
		for(int j=0;j<genome.size()/pattern_dimension;j++)
			pgenome[j]=genome[i*genome.size()/pattern_dimension+j];
		int redo=0;
		pstring[i]=printRandomProgram(pgenome,redo);
		if(redo>=REDO_MAX) return "";
		char str[10];
		sprintf(str,"f%d(x)=",i+1);
		ret+=str;
		ret+=pstring[i];
		ret+="\n";
	}
	return ret;
}

double 	NNprogram::fitness(vector<int> &genome)
{
	double value=0.0;
	if(pgenome.size()!=genome.size()/pattern_dimension)
		pgenome.resize(genome.size()/pattern_dimension);
	for(int i=0;i<pattern_dimension;i++)
	{
		for(int j=0;j<pgenome.size();j++)
			pgenome[j]=genome[i*genome.size()/pattern_dimension+j];
		int redo=0;
		pstring[i]=printRandomProgram(pgenome,redo);
		if(redo>=REDO_MAX) return -1e+100;
	}
	mapper->setExpr(pstring);
    extern int random_seed;
    srand48(random_seed);
	model->setPatternDimension(pattern_dimension);
	if(model_type==MODEL_NEURAL) model->randomizeWeights();
	value=model->train1();
	if(isnan(value) || isinf(value)) return -1e+100;
	if(fabs(value)>=1e+100) return -1e+100;
    if (neuralRulesMode) {
        // The neural error defines the primary fitness bucket. Within each
        // bucket, use rule error as the principal tie-breaker. Never allow a
        // candidate from a worse neural bucket to outrank a better bucket.
        const double tolerance = neuralRulesTolerance;
        const double bucket = std::floor(value / tolerance);
        if (!std::isfinite(bucket) || std::fabs(bucket) > 1e+12) return -1e+100;
        const double re = ruleError();
        if (!std::isfinite(re)) return -1e+100;
        const double fraction = std::max(0.0, std::min(1.0, value/tolerance-bucket));
        return -(bucket + 0.5*re + 0.49*fraction);
    }
    if (ruleWeight > 0.0) {
        const double re = ruleError();
        if (!std::isfinite(re)) return -1e+100;
        // Bound model loss to [0,1) before mixing with classification error.
        const double ml = std::max(0.0, value);
        return -((1.0-ruleWeight)*(ml/(1.0+ml)) + ruleWeight*re);
    }
	return -value;
}

void NNprogram::setRuleWeight(double w)
{
    ruleWeight = std::max(0.0, std::min(1.0, w));
}

void NNprogram::setNeuralRules(double tolerance)
{
    neuralRulesMode = true;
    neuralRulesTolerance = (std::isfinite(tolerance) && tolerance > 0.0)
        ? tolerance : 0.01;
    ruleWeight = 0.0;
}

// Fixed, non-evolutionary ordered IF/ELSE rule learner on constructed features.
// The first matching rule wins, as in QGenClass::ClassProgram::fitness().
// Each class (except the default class) gets one threshold rule.
// Rules are learned using TRAIN data only. No test labels are consulted.
double NNprogram::ruleError()
{
    const std::vector<Data> raw = ruleTrainSet->getAllXpoint();
    const Data labels = ruleTrainSet->getAllYPoints();
    if (raw.empty() || raw.size() != labels.size()) return 1.0;
    std::vector<Data> features;
    Data ys;
    features.reserve(raw.size());
    ys.reserve(raw.size());
    for (size_t i=0; i<raw.size(); ++i) {
        Data mapped(pattern_dimension, 0.0);
        Data x=raw[i];
        if (!mapper->map(x, mapped)) return 1.0;
        features.push_back(mapped);
        ys.push_back(labels[i]);
    }
    Data classes=ys;
    std::sort(classes.begin(),classes.end());
    classes.erase(std::unique(classes.begin(),classes.end()),classes.end());
    if (classes.size()<2) return 0.0;
    struct Rule {int feature; double threshold; bool less; double label;};
    std::vector<Rule> rules;
    // For each class, fit the best one-dimensional IF condition on
    // the samples not already covered by earlier rules.
    std::vector<char> active(ys.size(),1);
    for(size_t k=0;k+1<classes.size();++k) {
        const double target=classes[k];
        int bestCorrect=-1;
        Rule best={0,0.0,true,target};
        // Sorted sweep: O(N log N) per feature instead of O(N^2).
        for(int j=0;j<pattern_dimension;++j) {
            std::vector<std::pair<double,bool>> sorted;
            int positives=0, negatives=0;
            for(size_t i=0;i<ys.size();++i) if(active[i]) {
                double x=features[i][j];
                if (!std::isfinite(x)) return 1.0;
                bool positive=(ys[i]==target);
                sorted.emplace_back(x,positive);
                if(positive) ++positives; else ++negatives;
            }
            if(sorted.empty()) continue;
            std::sort(sorted.begin(),sorted.end(),
                      [](const std::pair<double,bool>& a,
                         const std::pair<double,bool>& b) {return a.first<b.first;});
            int leftPos=0, leftNeg=0;
            auto consider = [&](double threshold) {
                int correctLess=leftPos+(negatives-leftNeg);
                if(correctLess>bestCorrect) {
                    bestCorrect=correctLess;
                    best={j,threshold,true,target};
                }
                int correctGreater=leftNeg+(positives-leftPos);
                if(correctGreater>bestCorrect) {
                    bestCorrect=correctGreater;
                    best={j,threshold,false,target};
                }
            };
            consider(std::nextafter(sorted.front().first,
                                    -std::numeric_limits<double>::infinity()));
            size_t index=0;
            while(index<sorted.size()) {
                double current=sorted[index].first;
                do {
                    if(sorted[index].second) ++leftPos; else ++leftNeg;
                    ++index;
                } while(index<sorted.size() && sorted[index].first==current);
                double threshold=(index==sorted.size())
                    ? std::nextafter(current,std::numeric_limits<double>::infinity())
                    : current+(sorted[index].first-current)*0.5;
                consider(threshold);
            }
        }
        rules.push_back(best);
        for(size_t i=0;i<ys.size();++i) if(active[i]) {
            const bool hit=best.less ? features[i][best.feature]<=best.threshold
                                     : features[i][best.feature]>best.threshold;
            if(hit) active[i]=0;
        }
    }
    size_t errors=0;
    for(size_t i=0;i<ys.size();++i) {
        double predicted=classes.back();
        for(const Rule &r:rules) {
            const bool hit=r.less ? features[i][r.feature]<=r.threshold
                                  : features[i][r.feature]>r.threshold;
            if(hit) {predicted=r.label;break;}
        }
        if(predicted!=ys[i]) ++errors;
    }
    return double(errors)/ys.size();
}

Model	*NNprogram::getModel()
{
	return model;
}

Mapper	*NNprogram::getMapper()
{
	return mapper;
}

NNprogram::~NNprogram()
{
	delete program;
	delete model;
	delete mapper;
}

