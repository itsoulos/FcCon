#ifndef MKMEANS_H
#define MKMEANS_H
# include <CORE/dataset.h>

class Mkmeans
{
private:
    /**
      dataset: To dataset me ta dedomena
      center : to kentro kathe omadas
      member : gia kathe protypo se poia omada einai
      team   : to plithos omadon
    */
    vector<Data> center;
    vector<int>  member;
    int team;
    vector<int> teamMembers;
    vector<Data> xpoint;

    void initCenters();
    int  nearestTeam(Data &x);
    void updateCenters();
    double distance(Data &x,Data &y);
public:
    Mkmeans(vector<Data> &xx,int nteams);
    void    runAlgorithm();
    vector<Data> getCenters();
    Data          getVariances();
};

#endif // MKMEANS_H
