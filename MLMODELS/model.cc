# include <MLMODELS/model.h>
# include <stdlib.h>
# include <string.h>
# include <stdio.h>
# include <math.h>

//# define SCALEFACTOR

Matrix xmin,xmax,xmean,xstd,xcurrent;

double Model::distance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        sum += std::pow(a[i] - b[i], 2);
    }
    return std::sqrt(sum);
}



std::vector<int> Model::kNearest(
    const std::vector<Sample>& minority,
    int index,
    int k
    ) {
    std::vector<std::pair<double, int>> distances;

    for (size_t i = 0; i < minority.size(); i++) {
        if (i == index) continue;
        double d = distance(minority[index].features,
                            minority[i].features);
        distances.push_back({d, i});
    }

    std::sort(distances.begin(), distances.end());

    std::vector<int> neighbors;
    for (int i = 0; i < k && i < distances.size(); i++) {
        neighbors.push_back(distances[i].second);
    }

    return neighbors;
}

std::vector<Sample> Model::applySMOTE(
    const std::vector<Sample>& data,
    int k
    ) {
    std::map<double, std::vector<Sample>> classes;

    for (const auto& s : data) {
        classes[s.label].push_back(s);
    }

    // εύρεση πλειοψηφικής κλάσης
    int maxCount = 0;
    for (auto& c : classes) {
        maxCount = std::max(maxCount, (int)c.second.size());
    }

    std::vector<Sample> balanced = data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);

    for (auto& [label, samples] : classes) {
        if (samples.size() == maxCount) continue;

        int needed = maxCount - samples.size();
        int idx = 0;

        while (needed-- > 0) {
            const Sample& x = samples[idx % samples.size()];
            auto neighbors = kNearest(samples, idx % samples.size(), k);

            int nIndex = neighbors[gen() % neighbors.size()];
            const Sample& n = samples[nIndex];

            Sample synthetic;
            synthetic.label = label;

            for (size_t f = 0; f < x.features.size(); f++) {
                double gap = dist(gen);
                double val = x.features[f] +
                             gap * (n.features[f] - x.features[f]);
                synthetic.features.push_back(val);
            }

            balanced.push_back(synthetic);
            idx++;
        }
    }

    return balanced;
}


void    Model::enableSmote()
{
    vector<Sample> data;
    int count = origx.size();
    data.resize(count);
    for(int i=0;i<count;i++)
    {
        data[i].features=origx[i];
        data[i].label=origy[i];
    }

    vector<Sample> data2=applySMOTE(data,5);
    origx.clear();
    origy.clear();
    count = data2.size();
    origx.resize(count);
    origy.resize(count);
    xpoint.resize(count);
    ypoint.resize(count);

    for(int i=0;i<count;i++)
    {
        origx[i]=data2[i].features;
	origy[i]=data2[i].label;
	ypoint[i]=data2[i].label;
    }
}

static int nearestClassIndex(vector<double> &dclass,double value)
{
    int pos=-1;
    double dmin=1e+100;
    for(int i=0;i<dclass.size();i++)
    {
        double d = fabs(dclass[i]-value);
        if(d<dmin)
        {
            dmin = d;
            pos = i;
        }
    }
    return pos;
}
static int isone(double x)
{
	return fabs(x-1.0)<1e-5;
}

static int iszero(double x)
{
	return fabs(x)<1e-5;
}

double scale_factor = 1.0;

Model::Model(Mapper *m)
{
	num_weights = 1;
	pattern_dimension = 0;
	mapper = m;
	isvalidation=0;
}

Matrix	Model::getWeights()
{
	return weight;
}


int	Model::getOriginalDimension() const
{
	return original_dimension;
}


int	Model::getNumPatterns() const
{
	return ypoint.size();
}

void	Model::randomizeWeights()
{
	weight.resize((pattern_dimension+2)*num_weights);
	setDimension(weight.size());
	for(int i=0;i<weight.size();i++) weight[i]=0.1*(2.0*drand48()-1.0);
}

void	Model::setPatternDimension(int d)
{
	if(pattern_dimension!=d)
	{
		pattern_dimension = d;
		for(int i=0;i<xpoint.size();i++) 
			xpoint[i].resize(pattern_dimension);
	}
	xcurrent.resize(d);
}

void 	Model::readPatterns(char *filename)
{
	FILE *fp;
	fp = fopen(filename,"r");
	if(!fp) return;
	int count,d;
	fscanf(fp,"%d",&d);
	if(d<=0) 
	{
		d=0;
		fclose(fp);
		return;
	}
	original_dimension =d ;
	fscanf(fp,"%d",&count);
	if(count<=0)
	{
		d=0;
		fclose(fp);
		return;
	}
	origx.resize(count);
	xpoint.resize(count);
	origy.resize(count);
	ypoint.resize(count);
	xmax.resize(d);
	xmin.resize(d);
	xmean.resize(d);
	xstd.resize(d);
	for(int i=0;i<d;i++)
	{
		xmax[i]=-1e+100;
		xmin[i]= 1e+100;
		xmean[i]=0.0;
		xstd[i]=0.0;
	}
	int count1=0,count2=0,two_classes_flag=1;
	for(int i=0;i<count;i++)
	{
		origx[i].resize(d);
		for(int j=0;j<d;j++)
		{
			fscanf(fp,"%lf",&origx[i][j]);
			if(origx[i][j]>xmax[j]) xmax[j]=origx[i][j];
			if(origx[i][j]<xmin[j]) xmin[j]=origx[i][j];
			xmean[j]+=origx[i][j];
			xstd[j]+=origx[i][j]*origx[i][j];
		}
		fscanf(fp,"%lf",&ypoint[i]);
		if(iszero(ypoint[i])) count1++;
		else 
		if(isone(ypoint[i])) count2++;
		else	two_classes_flag=0;
		origy[i]=ypoint[i];
        int found=-1;
        for(int j=0;j<dclass.size();j++)
        {
            if(fabs(dclass[j]-ypoint[i])<1e-5)
            {
                found=j;
                break;
            }
        }
        if(found==-1)
            dclass.push_back(ypoint[i]);
	}
	fclose(fp);

    //sort dclass
    for(int i=0;i<dclass.size();i++)
    {
        for(int j=0;j<dclass.size()-1;j++)
        {
            if(dclass[j+1]<dclass[j])
            {
                double t = dclass[j];
                dclass[j]=dclass[j+1];
                dclass[j+1]=t;
            }
        }
    }
	if(two_classes_flag) scale_factor=count1*1.0/count2;
    extern bool fc_enablesmote;
    if(fc_enablesmote)
        enableSmote();
}

void	Model::transform(Matrix x,Matrix &x1)
{
	for(int  i=0;i<x.size();i++) 
	//x1[i]=sig(x[i]);
		//x1[i]=(x[i]-xmean[i])/(xstd[i]);
	x1[i]=x[i];
	//x1[i]=(x[i]-xmin[i])/(xmax[i]-xmin[i]);
}

void	Model::setNumOfWeights(int w)
{
	num_weights = w;
}

Matrix	Model::getXpoint(int pos)
{
	return xpoint[pos];
}

double  Model::getYPoint(int pos)
{
	return ypoint[pos];
}

double  Model::getModelAtPoint(int pos)
{
	double v = output(xpoint[pos]);
	return v;
}

int	Model::getPatternDimension() const
{
	return pattern_dimension;
}

double	Model::valError()
{
	double s=0.0;
	for(int i=4*xpoint.size()/5;i<xpoint.size();i++)
	{
		double v = output(xpoint[i]);
		s+=(v-ypoint[i])*(v-ypoint[i]);
	}
	return s;
}

double  Model::getAverageClassError(Matrix &x)
{
    if(weight.size()!=x.size()) weight.resize(x.size());
    for(int i=0;i<x.size();i++) weight[i] = x[i];
    double sum = 0.0;
    int end=xpoint.size();
    if(isvalidation) end=4*xpoint.size()/5;
    vector<int> missed,belong;
    missed.resize(dclass.size());
    belong.resize(dclass.size());
    for(int i=0;i<(int)missed.size();i++)
    {
        missed[i]=0;
        belong[i]=0;
    }
    for(int i=0;i<end;i++)
    {
        double v = output(xpoint[i]);
        double c1 = nearestClassIndex(dclass,v);
        double c2 = nearestClassIndex(dclass,ypoint[i]);
        if(fabs(c1-c2)>1e-5)
            missed[(int)c2]++;
        belong[(int)c2]++;
    }

    for(int i=0;i<(int)missed.size();i++)
    {
        double dc = missed[i]*100.0/belong[i];
        sum+=dc;
    }
    return sum/dclass.size();
}

double	Model::funmin(Matrix x)
{
    extern bool fc_balanceclass;
    extern bool fc_enablemean;

    if(fc_balanceclass)
        return getAverageClassError(x);

	if(weight.size()!=x.size()) weight.resize(x.size());
	for(int i=0;i<x.size();i++) weight[i] = x[i];


    if(fc_enablemean)
    {

        double avg_precision,avg_recall,avg_fscore;
        getPrecisionRecall(
            avg_precision,
            avg_recall,
            avg_fscore);

        double d = 100.0*(1.0-sqrt(avg_precision * avg_recall));

        return d;
    }
	double s=0.0;
	int end=xpoint.size();
	if(isvalidation) end=4*xpoint.size()/5;
	int correct1=0;
	int correct2=0;
	int count1=0,count2=0;
	for(int i=0;i<end;i++)
	{
		double v = output(xpoint[i]);
		double e=v-ypoint[i];
		e=e*e;
#ifdef SCALEFACTOR
		if(!isone(scale_factor))
		{
			if(scale_factor>1) 
			{
				if(isone(ypoint[i])) e=e*scale_factor;
			}
			else
			{
				if(iszero(ypoint[i])) e=e*1.0/scale_factor;
			}
		}
#endif
		if(isnan(v) || isinf(v)) return 1e+100;
		s+=e;
		if(isnan(s) || isinf(s)) return 1e+100;
	}
	return s;
}

void  Model::granal(Matrix x,Matrix &g)
{
	if(weight.size()!=x.size())
	weight.resize(x.size());
	for(int i=0;i<x.size();i++) 
	{
		weight[i] = x[i];
		g[i]=0.0;
	}
	double s=0.0;
	Matrix gtemp;
	gtemp.resize(g.size());
	int end=xpoint.size();
	if(isvalidation) end=4*xpoint.size()/5;
	for(int i=0;i<end;i++)
	{
		double	e=output(xpoint[i])-ypoint[i];
#ifdef SCALEFACTOR
		if(!isone(scale_factor))
		{
			if(scale_factor>1) 
			{
				if(isone(ypoint[i])) e=e*scale_factor;
			}
			else
			{
				if(iszero(ypoint[i])) e=e*1.0/scale_factor;
			}
		}
#endif
		getDeriv(xpoint[i],gtemp);
		for(int j=0;j<g.size();j++)
		{
			g[j]+=2.0*e*gtemp[j];
		}
	}
}

/*	Grafei to apotelemsa tou mapping 
 *	sto arxeio train. Diabazei apo to arxeio itest
 *	ta test patterns kai ta grafei sto arxeio otest.
 *	O skopos einai na xrisimopoiithoyn ta patterns 
 *	gia epexergasia apo kapoio montelo poy den 
 *	kalyptetai sto paketo.
 * */
void	Model::print(char *train,char *itest, char *otest)
{
	FILE *fp=fopen(train,"w");
	if(!fp) return;
	fprintf(fp,"%d\n%d\n",pattern_dimension,origx.size());
	for(int i=0;i<origx.size();i++)
	{
		mapper->map(origx[i],xpoint[i]);
		for(int j=0;j<pattern_dimension;j++)
			fprintf(fp,"%lf ",xpoint[i][j]);
		fprintf(fp,"%lf\n",ypoint[i]);
	}
	fclose(fp);
	FILE *fin=fopen(itest,"r");
	if(!fin) return;
	FILE *fout=fopen(otest,"w");
	int d,count;
	fscanf(fin,"%d",&d);
	fscanf(fin,"%d",&count);
	Matrix testx;
	testx.resize(d);
	double testy;
	Matrix xx;
	xx.resize(pattern_dimension);
	fprintf(fout,"%d\n%d\n",pattern_dimension,count);
	for(int i=0;i<count;i++)
	{
		for(int j=0;j<d;j++)
			fscanf(fin,"%lf",&testx[j]);
		fscanf(fin,"%lf",&testy);
		mapper->map(testx,xx);
		for(int j=0;j<pattern_dimension;j++)
			fprintf(fout,"%lf ",xx[j]);
		fprintf(fout,"%lf\n",testy);
	}
	fclose(fin);
	fclose(fout);
}

double	Model::testError(char *filename)
{
	double testy;
	Matrix testx;
	int count;
	int dim;
	FILE *fp;
	fp=fopen(filename,"r");
	if(!fp) return -1.0;
	fscanf(fp,"%d",&dim);
	if(dim<=0) 
	{
		fclose(fp);
		return -1.0;
	}	
	fscanf(fp,"%d",&count);
	if(count<=0)
	{
		fclose(fp);
		return -1.0;
	}
	testx.resize(pattern_dimension);
	Matrix xx;
	xx.resize(dim);
	double sum = 0.0;
	Matrix xx2;
	xx2.resize(dim);
	for(int i=0;i<count;i++)
	{
		for(int j=0;j<dim;j++) fscanf(fp,"%lf",&xx[j]);
		transform(xx,xx2);
		mapper->map(xx2,testx);
		fscanf(fp,"%lf",&testy);
		double d=output(testx);
		sum+=pow(d-testy,2.0);
	}
	fclose(fp);
	return (sum);
}

double	Model::classTestError(char *filename,double &precision,double &recall)
{
	vector<double> classes;
	Matrix testx;
	double testy;
	int count;
	int dim;
	FILE *Fp;
	int   tp=0,fp=0,tn=0,fn=0;
	precision = 0.0;
	recall    = 0.0;
	Fp=fopen(filename,"r");
	if(!Fp) return -1.0;
	fscanf(Fp,"%d",&dim);
	if(dim<=0) 
	{
		fclose(Fp);
		return -1.0;
	}	
	fscanf(Fp,"%d",&count);
	if(count<=0)
	{
		fclose(Fp);
		return -1.0;
	}

	double x1,y1;
	for(int i=0;i<count;i++)
	{
		for(int j=0;j<dim;j++)
		{
			fscanf(Fp,"%lf",&x1);
		}
		fscanf(Fp,"%lf",&y1);
		int found=-1;
		for(int j=0;j<classes.size();j++)
		{
			if(fabs(classes[j]-y1)<1e-8)
			{
				found = j;
				break;
			}
		}
		if(found==-1)
		{
			int s=classes.size();
			classes.resize(s+1);
			classes[s]=y1;
		}
	}
	fclose(Fp);
	
	Fp=fopen(filename,"r");
	fscanf(Fp,"%d",&dim);
	fscanf(Fp,"%d",&count);
	testx.resize(pattern_dimension);
	Matrix xx;
	xx.resize(dim);
	double sum = 0.0;
	Matrix xx2;
	xx2.resize(dim);
	int count1=0,count2=0,est1=0,est2=0;
	for(int i=0;i<count;i++)
	{
		for(int j=0;j<dim;j++) 
		{
			fscanf(Fp,"%lf",&xx[j]);
			/**/
			/**/
		}
		mapper->map(xx,testx);
		fscanf(Fp,"%lf",&testy);
		double c=output(testx);
		
		int found =-1;
		double dmin=1e+10;
		for(int j=0;j<classes.size();j++)
			if(fabs(classes[j]-c)<dmin)
			{
				found=j;
				dmin=fabs(classes[j]-c);
			}
		if(classes.size()==2)
		{
			if(isone(classes[found]) && isone(testy)) tp++;
			else
			if(isone(classes[found]) && iszero(testy)) fp++;
			else
			if(iszero(classes[found]) && iszero(testy)) tn++;
			else
			if(iszero(classes[found]) && isone(testy)) fn++;
			else ;
	//			printf("unspecified %lf %lf \n",classes[found],testy);
		}
		double myclass=classes[found];
		if(fabs(testy)<1e-5) count1++; else count2++;
		if(fabs(testy-classes[found])<1e-5)
		{
			if(fabs(classes[found])<1e-5) est1++; else est2++;
		}
		sum+=(fabs(testy-myclass)>1e-5);
	}
	fclose(Fp);
//	printf("CLASS1 = %2.lf%% CLASS2=%.2lf%%\n",est1*100.0/count1,est2*100.0/count2);
	recall=tp*1.0/(tp*1.0+fn*1.0);
	precision=tp*1.0/(tp*1.0+fp*1.0);
	return (sum)/count;
}


void	Model::printConfusionMatrix(
                                 vector<double> &dclass,
                                 vector<double> &T,vector<double> &O,
                                 vector<double> &precision,
                                 vector<double> &recall)
{
    int i,j;
    int N=T.size();
    int nclass=dclass.size();
    precision.resize(nclass);
    recall.resize(nclass);
    int **CM;
    //printf("** CONFUSION MATRIX ** Number of classes: %d\n",nclass);
    CM=new int*[nclass];
    for(i=0;i<nclass;i++) CM[i]=new int[nclass];
    for(i=0;i<nclass;i++)
        for(j=0;j<nclass;j++) CM[i][j] = 0;

    for(i=0;i<N;i++) CM[(int)T[i]][(int)O[i]]++;
    for(i=0;i<nclass;i++)
    {
        double sum = 0.0;
        for(j=0;j<nclass;j++)
            sum+=CM[j][i];
        precision[i]=sum==0?-1:CM[i][i]/sum;
        sum = 0.0;
        for(j=0;j<nclass;j++)
            sum+=CM[i][j];
        recall[i]=sum==0?-1:CM[i][i]/sum;
    }
    for(i=0;i<nclass;i++)
    {
        for(j=0;j<nclass;j++)
        {
            //printf("%4d ",CM[i][j]);
        }
        //printf("\n");
        delete[] CM[i];
    }
    delete[] CM;
}

void    Model::getPrecisionRecall(
    double &avg_precision,
    double &avg_recall,
    double &avg_fscore)
{
    int count = xpoint.size();
    vector<double> T;
    vector<double> O;
    T.resize(count);
    O.resize(count);


    for(unsigned int i=0;i<count;i++)
    {
        //mapper->map(xpoint[i],xx);
        double tempOut = output(xpoint[i]);
        T[i]=nearestClassIndex(dclass,ypoint[i]);
        O[i]=nearestClassIndex(dclass,tempOut);
    }

    vector<double> precision;
    vector<double> recall;
    vector<double> fscore;
    fscore.resize(dclass.size());
    avg_precision = 0.0, avg_recall = 0.0,avg_fscore=0.0;
    printConfusionMatrix(dclass,T,O,precision,recall);
    int icount1=dclass.size();
    int icount2=dclass.size();
    for(int i=0;i<dclass.size();i++)
    {
        if(precision[i]>=0)
            avg_precision+=precision[i];
        else icount1--;
        if(recall[i]>=0)
            avg_recall+=recall[i];
        else icount2--;
        fscore[i]=2.0*precision[i]*recall[i]/(precision[i]+recall[i]);
        avg_fscore+=fscore[i];
    }
    avg_precision/=icount1;
    avg_recall/=icount2;
    avg_fscore=2.0 * avg_precision * avg_recall/(avg_precision+avg_recall);

}

void    Model::getPrecisionRecall(
                               const char *filename,
                               double &avg_precision,
                               double &avg_recall,
                               double &avg_fscore)
{

    int count;
    int dim;
    FILE *Fp=fopen(filename,"r");
     fscanf(Fp,"%d",&dim);
    fscanf(Fp,"%d",&count);
    vector<double> dclass;
    vector<vector<double>> testx;
    vector<double>  testy;
    testx.resize(count);
    testy.resize(count);
    for(int i=0;i<count;i++)
    {
        testx[i].resize(dim);
        for(int j=0;j<dim;j++)
            fscanf(Fp,"%lf",&testx[i][j]);
        fscanf(Fp,"%lf",&testy[i]);
        int found =-1;
        for(int j=0;j<dclass.size();j++)
        {
            if(fabs(dclass[j]-testy[i])<1e-5)
            {
                found=j;
                break;
            }
        }
        if(found==-1)
        {
            dclass.push_back(testy[i]);
        }
    }
    fclose(Fp);


    for(int i=0;i<dclass.size();i++)
    {
        for(int j=0;j<dclass.size()-1;j++)
        {
            if(dclass[j+1]<dclass[j])
            {
                double t = dclass[j];
                dclass[j]=dclass[j+1];
                dclass[j+1]=t;
            }
        }
    }



    vector<double> T;
    vector<double> O;
    T.resize(count);
    O.resize(count);

    vector<double> xx;
    xx.resize(pattern_dimension);

    for(unsigned int i=0;i<count;i++)
    {
        mapper->map(testx[i],xx);
        double tempOut = output(xx);
        T[i]=nearestClassIndex(dclass,testy[i]);
        O[i]=nearestClassIndex(dclass,tempOut);
    }

    vector<double> precision;
    vector<double> recall;
    vector<double> fscore;
    fscore.resize(dclass.size());
    avg_precision = 0.0, avg_recall = 0.0,avg_fscore=0.0;
    printConfusionMatrix(dclass,T,O,precision,recall);
    int icount1=dclass.size();
    int icount2=dclass.size();
    for(int i=0;i<dclass.size();i++)
    {
        if(precision[i]>=0)
            avg_precision+=precision[i];
        else icount1--;
        if(recall[i]>=0)
            avg_recall+=recall[i];
        else icount2--;
        fscore[i]=2.0*precision[i]*recall[i]/(precision[i]+recall[i]);
        avg_fscore+=fscore[i];
    }
    avg_precision/=icount1;
    avg_recall/=icount2;
    avg_fscore=2.0 * avg_precision * avg_recall/(avg_precision+avg_recall);

}

void	Model::enableValidation()
{
	isvalidation=1;
}

Model::~Model()
{
}
