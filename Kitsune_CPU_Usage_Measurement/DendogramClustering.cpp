#include "DendogramClustering.h"
#include <algorithm>


DendogramClustering::DendogramClustering()
{
}



DendogramClustering::~DendogramClustering()
{

}
double ** transpose(double **matrix, int rows, int columns) {
	double ** trans;
	trans = new double *[columns];
	for (int i = 0; i<columns; i++) {
		trans[i] = new double[rows];
		for (int j = 0; j<rows; j++)
			trans[i][j] = matrix[j][i];
	}
	return trans;
	for (int i = 0; i<columns; i++)
		delete[] trans[i];
	delete[] trans;
}
void DendogramClustering::clusterUsingIncrementalFuzzyDendogram(double ** data, int rows, int cols, int maxSizeOfCluster = 10, int maxFuz = 3)
{


	data = transpose(data, rows, cols);
	vector<vector<vector<double>>> clusters;
	vector<double> centroids;
	vector<vector<int>> featuresIndexes;
	for (int i = 0; i < cols; i++)
	{

		vector<double> rec;
		copy(rec.begin(), rec.end(), data[i]);

		vector<vector<double>> group; (1,0);

		copy(group[0].begin(), group[0].end(), data[i]);
		clusters.push_back(group);

		vector<int> groupIdx;
		groupIdx.push_back(i);
		featuresIndexes.push_back(groupIdx);



		vector<double> a(group[0].size(), 0);
		for (int i = 0; i < group.size(); i++)
		{
			std::transform(a.begin(), a.end(), group[i].begin(), a.begin(), std::plus<int>());
		}

		centroids.push_back(std::accumulate(a.begin(),a.end(), 0.0) / a.size());
	}


	int actualSizeOfCluster = 0;

	for (int k = 0; k < clusters.size(); k++)
		if (actualSizeOfCluster > clusters[k].size())
			actualSizeOfCluster = clusters[k].size();

	int minC1;
	int minC2;
	double c1c2Diff;
	while (actualSizeOfCluster < maxSizeOfCluster) {
		double minDist = 10000000000000000;
		vector<double> distAr;
		vector<int> indexesAr;
		for (int c1 = 0; c1 < centroids.size(); c1++)
		{
			for (int c2 = 0; c2 < centroids.size(); c2++) {
				if (c1 != c2) {

					c1c2Diff = std::abs(centroids[c1] - centroids[c2]);



					if (c1c2Diff < minDist)
					{

						minDist = c1c2Diff;
						 minC1 = c1;
						 minC2 = c2;
					}
				}
			}
		}

		for (int c2 = 0; c2 < centroids.size(); c2++)
		{
			if (c2 != minC1)
				c1c2Diff = std::abs(centroids[minC1] - centroids[c2]);
			distAr.push_back(c1c2Diff);
			indexesAr.push_back(c2);

		}









				typedef std::vector<int> int_vec_t;
		typedef std::vector<double> dist_vec_t;
		typedef std::vector<size_t> index_vec_t;

		class SequenceGen {
		  public:
			SequenceGen (int start = 0) : current(start) { }
			int operator() () { return current++; }
		  private:
			int current;
		};

		class Comp{
			int_vec_t& _v;
		  public:
			Comp(int_vec_t& v) : _v(v) {}
			bool operator()(size_t i, size_t j){
				 return _v[i] < _v[j];
		   }
		};

		index_vec_t indices(indexesAr.size());
		std::generate(indices.begin(), indices.end(), SequenceGen(0));
		//indices are {0, 1, 2}

		int_vec_t Index = indexesAr;
		dist_vec_t Values = distAr;

		std::sort(indices.begin(), indices.end(), Comp(Index));
		//now indices are {1,2,0}



		std::sort(indexesAr.begin(), indexesAr.end(),
			[&distAr](const auto& i, const auto& j) { return distAr[i] > distAr[j]; });


			//indexesAr = [x for _, x in sorted(zip(distAr, indexesAr))];

					for (int idx = 0; idx < maxFuz; idx++)
					{

						if (idx < maxFuz - 1)
							mergeClusters(clusters,featuresIndexes, centroids, minC1, indexesAr[idx], false);
						else
							mergeClusters(clusters, featuresIndexes, centroids, minC1, indexesAr[idx], true);
					}
					actualSizeOfCluster = 0;
					for (int f = 0; f < clusters.size(); f++)
					{
						actualSizeOfCluster = std::max((int)actualSizeOfCluster, (int)clusters[f].size());
					}
						
}
		cout << "finished clustering..." << endl;

}
void DendogramClustering::mergeClusters(vector <vector<vector<double>>> clusters,vector<vector<int>> featuresIndexes, vector<double> centroids, int c1, int c2, bool isLast = false)
{

	clusters[c1].insert(clusters[c1].end(), clusters[c2].begin(), clusters[c2].end());
	featuresIndexes[c1].insert(featuresIndexes[c1].end(), featuresIndexes[c2].begin(), featuresIndexes[c2].end());

	vector<double> a(clusters[c1][0].size(), 0);
	for (int i = 0; i < clusters[c1].size(); i++)
	{
		std::transform(a.begin(), a.end(), clusters[c1][i].begin(), a.begin(), std::plus<int>());
	}


	



		//centroids[c1] = std::accumulate(clusters[c1].begin(),clusters[c2].end(), 0.0) / clusters[c1].size();
	centroids[c1] = std::accumulate(a.begin(), a.end(), 0.0) / a.size();
		
	
		if (isLast == true)
		{
			clusters.erase(clusters.begin() + c2);
			centroids.erase(centroids.begin() + c2);
		}
			
}


