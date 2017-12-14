#ifndef MODELEX_H
#define MODELEX_H
#include <string>
#include <iostream>
#include <sstream>

#include <fstream>
#include <iterator>
#include <vector>
#define  D_SCL_SECURE_NO_WARNINGS


#define _CRT_SECURE_NO_WARNINGS
using namespace std;
class modelEx
{
public:
	modelEx::modelEx() {}
	vector<vector<double> > modelEx::loadRow(ifstream & myfile);
 vector<vector<double> > modelEx::loadDataset(const char * path);
};

#endif // MODELEX_H
