# include <GE/rule.h>
# include <iostream>
Rule::Rule()
{
	length =0;
	weight =  1.0;
}

void	Rule::addSymbol(Symbol *s)
{
	data.push_back(s);
	length++;
}

int	Rule::getSymbolPos(string s)
{
	for(int i=0;i<length;i++)
		if(data[i]->getName()==s) return i;
	return -1;
}

Symbol	*Rule::getSymbol(int pos) const
{
	if(pos<0 || pos>=length) return NULL;
	return data[pos];
}

void	Rule::setSymbol(int pos,Symbol *s)
{
	if(pos<0 || pos>=length) return; 
	data[pos]=s;
}

int	Rule::getLength() const
{
	return length;
}

double	Rule::getWeight() const
{
	return weight;
}

void	Rule::setWeight(double w)
{
	weight = w;
}

void	Rule::updatePositive(vector<int> genome,int &pos,int &redo,
		vector<Rule*> &rule,vector<double> &weight)
{
#ifdef STOCHASTIC
	const double epsilon=0.01;
	string str="";
	string str2="";
	Rule *r;
	int old_pos;
	for(int i=0;i<length;i++)
	{
		Symbol *s=data[i];
		if(s->getTerminalStatus())
			str=str+s->getName();	
		else
		{
			if(pos>=genome.size()) {redo++;pos=0;}
			int rulepos;
			double rand=drand48();
			if(rand<=s->getRule(genome[pos]%
					s->getCountRules())->getWeight())
				rulepos=genome[pos]%s->getCountRules();
			else
			{
				rulepos = (int)(drand48()*(s->getCountRules()-1));
			}
			r=s->getRule(rulepos);
			int found=-1;
			for(int j=0;j<rule.size();j++)
				if(rule[j]==r)
				{
					found=j;
					break;
				}
			if(found==-1)
			{
				int k=rule.size();
				rule.resize(k+1);
				weight.resize(k+1);
				rule[k]=r;
				weight[k]=r->getWeight()*(1.0+epsilon);
			}
			pos++;
			if(pos>=genome.size()) {redo++;pos=0;}
			if(redo>=REDO_MAX) return ;
			r->updatePositive(genome,pos,redo,rule,weight);
		}
	}
#endif
}

void	Rule::updateNegative(vector<int> genome,int &pos,int &redo,
		vector<Rule*> &rule,vector<double> &weight)
{
#ifdef STOCHASTIC
	const double epsilon=0.01;
	string str="";
	string str2="";
	Rule *r;
	int old_pos;
	for(int i=0;i<length;i++)
	{
		Symbol *s=data[i];
		if(s->getTerminalStatus())
		{
			str=str+s->getName();	
		}
		else
		{
			if(pos>=genome.size()) {redo++;pos=0;}
			int rulepos;
			if(drand48()<=s->getRule(genome[pos]%
					s->getCountRules())->getWeight())
				rulepos=genome[pos]%s->getCountRules();
			else
			{
				rulepos = (int)(drand48()*(s->getCountRules()-1));
			}
			r=s->getRule(rulepos);
			int found=-1;
			for(int j=0;j<rule.size();j++)
				if(rule[j]==r)
				{
					found=j;
					break;
				}
			if(found==-1)
			{
				int k=rule.size();
				rule.resize(k+1);
				weight.resize(k+1);
				rule[k]=r;
				weight[k]=r->getWeight()/(1.0+epsilon);
			}
			pos++;
			if(pos>=genome.size()) {redo++;pos=0;}
			if(redo>=REDO_MAX) return ;
			r->updateNegative(genome,pos,redo,rule,weight);
		}
	}
#endif
}

string	Rule::printRule(vector<int> genome,int &pos,int &redo)
{
	string str="";
	string str2="";
	Rule *r;
	for(int i=0;i<length;i++)
	{
		Symbol *s=data[i];
		if(s->getTerminalStatus())
		{
			str=str+s->getName();	
		}
		else
		{
			if(pos>=genome.size()) {redo++;pos=0;}
#ifndef STOCHASTIC
			r=s->getRule((genome[pos]%s->getCountRules()));
#else
			int rulepos;
			double rand=drand48();
			if(rand<=s->getRule(genome[pos]%
					s->getCountRules())->getWeight())
				rulepos=genome[pos]%s->getCountRules();
			else
			{
				rulepos = (int)(drand48()*(s->getCountRules()-1));
			}
			r=s->getRule(rulepos);
#endif
			pos++;
			if(pos>=genome.size()) {redo++;pos=0;}
			if(redo>=REDO_MAX) return str;
			str2=r->printRule(genome,pos,redo);
			str=str+str2;
		}
	}
	return str;
}

Rule::~Rule()
{
}

