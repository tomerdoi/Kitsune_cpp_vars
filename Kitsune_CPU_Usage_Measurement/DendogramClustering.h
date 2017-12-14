#pragma once
#include "D:/thesis_data/code/integrated_code_kitsune/dA.h"
#include <numeric>
#include <math.h>
using namespace std;
class DendogramClustering
{
public:
	DendogramClustering();
	void clusterUsingIncrementalFuzzyDendogram(double ** data, int rows, int cols, int maxSizeOfCluster , int maxFuz );
	void DendogramClustering::mergeClusters(vector <vector<vector<double>>> clusters,vector<vector<int>> featuresIndexes, vector<double> centroids, int c1, int c2, bool isLast );
	

	~DendogramClustering();
};

