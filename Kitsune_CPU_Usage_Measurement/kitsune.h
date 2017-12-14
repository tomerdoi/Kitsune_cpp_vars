#ifndef KITSUNETEST_H
#define KITSUNETEST_H
#include <map>
#include <set>
#include "dA.h"
#include <fstream>
#define  D_SCL_SECURE_NO_WARNINGS

#define _CRT_SECURE_NO_WARNINGS
struct scores_arrays {
    double  **s1;
    double *s2;
};
class kitsune
{
public:
    kitsune();
    scores_arrays* KitsuneTest(vector<vector<double> > X,vector<int> map,int trainEnd);
    vector<int> convertMappingFileToVector (string path);
    scores_arrays* KitsuneTestExecute(vector<vector<double> > X,vector<int> mapIdx, vector<dA*> & ensLayer,dA & outputLayer,map < int , vector<int> > aeMap);
    scores_arrays* KitsuneTestTrain(vector<vector<double> > X,vector<int> mapIdx, vector<dA*> & ensLayer,dA & outputLayer,map < int , vector<int> > aeMap );
};

#endif // KITSUNETEST_H

