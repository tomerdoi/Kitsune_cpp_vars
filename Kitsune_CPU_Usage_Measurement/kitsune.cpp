#include "kitsune.h"

kitsune::kitsune()
{

}
scores_arrays* kitsune::KitsuneTest(vector<vector<double> > X, vector<int> mapIdx, int trainEnd) {
	//######### INIT ##########
	cout << "init" << endl;
	//# construct ensemble layer
	//aeIDs = np.unique(map)
	set<int> aeIDs;

	for (int i = 0; i<mapIdx.size(); i++)
		aeIDs.insert(mapIdx[i]);

	map < int, vector<int> > aeMap;
	//   for id in aeIDs:
	//       aeMap.append(np.where(map==id)[0])
	for (int i = 0; i<mapIdx.size(); i++)
		aeMap[mapIdx[i]].push_back(i);
	vector<dA*> ensLayer;
	dA * d;
	for (int i = 0; i<aeMap.size(); i++)
	{
		dA_params *params = new dA_params(aeMap[i].size(), 0, 0.01, 0, 0, 0.7);
		if (aeMap[i].size() == 0)
			continue;
		d = new dA(*params);
		ensLayer.push_back(d);
	}

	//# construct output layer
	dA_params params(aeIDs.size(), 0, 0.01, 0, 0, 0.7);
	dA outputLayer(params);

	//#init output variables
	double ** S_l1 = new double*[X.size()];
	for (int i = 0; i<X.size(); i++)
		S_l1[i] = new double[ensLayer.size()];
	double *S_l2 = new double[X.size()];

	//########## TRAIN ##########
	cout << "train" << endl;
	for (int i = 0; i<trainEnd; i++)
	{
		if (i % 1000 == 0)
			cout << i << endl;
		vector<double> x = X[i];
		//## EnsLayer
		for (int j = 0; j<ensLayer.size(); j++)
		{

			// #make sub inst
			vector<double> xi;
			for (int k = 0; k<aeMap[j].size(); k++)
				xi.push_back(x[aeMap[j][k]]);
			if (xi.size() == 0)
				continue;
			//x[aeMap[ensLayer[j]]];
			S_l1[i][j] = ensLayer[j]->train(xi)[0];
		}
		//## OutputLayer
		vector<double> v;
		//std::copy(v.begin(), v.end(), S_l1[i]);
		for (int r = 0; r < ensLayer.size(); r++)
			v.push_back(S_l1[0][r]);
		if (v.size() == 0)
			continue;
		S_l2[i] = outputLayer.train(v)[0];
	}

	//########## EXECUTE ##########
	cout << "execute" << endl;
	for (int i = trainEnd; i<X.size(); i++)
	{
		if (i % 1000 == 0)
			cout << i << endl;
		vector<double> x = X[i];
		//## EnsLayer
		for (int j = 0; j<ensLayer.size(); j++)
		{

			//# make sub inst
			// vector <double> xi = x[aeMap[j]];
			vector<double> xi;
			for (int k = 0; k<aeMap[j].size(); k++)

				xi.push_back(x[aeMap[j][k]]);
			if (xi.size() == 0)
				continue;
			S_l1[i][j] = ensLayer[j]->execute_r(xi)[0];
		}
		//## OutputLayer
		vector<double> v;
		//std::copy(v.begin(), v.end(), S_l1[i]);
		for (int r = 0; r < ensLayer.size(); r++)
			v.push_back(S_l1[0][r]);
		if (v.size() == 0)
			continue;
		S_l2[i] = outputLayer.execute_r(v)[0];
	}
	scores_arrays *sa = (scores_arrays*)malloc(8);
	sa->s1 = S_l1;
	sa->s2 = S_l2;
	return sa;
}

scores_arrays* kitsune::KitsuneTestExecute(vector<vector<double> > X, vector<int> mapIdx, vector<dA*> & ensLayer, dA & outputLayer, map < int, vector<int> > aeMap) {
	//######### INIT ##########
	// cout<<"init"<<endl;
	//# construct ensemble layer
	//aeIDs = np.unique(map)


	//#init output variables
	double ** S_l1 = new double*[X.size()];
	for (int i = 0; i<X.size(); i++)
		S_l1[i] = new double[ensLayer.size()];
	double *S_l2 = new double[X.size()];




	//########## EXECUTE ##########
	//cout<<"execute"<<endl;
	for (int i = 0; i<X.size(); i++)
	{
		// if (i % 1000 == 0)
		//   cout<<i<<endl;
		vector<double> x = X[i];
		//## EnsLayer

		for (int j = 0; j<ensLayer.size(); j++)
		{
			//cout<<" is the score eredfdffffffffffffff"<<endl;
			//# make sub inst
			// vector <double> xi = x[aeMap[j]];
			vector<double> xi;
			for (int k = 0; k<aeMap[j].size(); k++)

				xi.push_back(x[aeMap[j][k]]);
			if (xi.size() == 0)
				continue;
			S_l1[i][j] = ensLayer[j]->execute_r(xi)[0];
			//cout<<" is the score "<<S_l1[i][ j]<<endl;
		}
		//## OutputLayer
		vector<double> v;
		//std::copy(v.begin(), v.end(), S_l1[i]);
		for (int r = 0; r < ensLayer.size(); r++)
			v.push_back(S_l1[0][r]);
		for (int f = 0; f<ensLayer.size(); f++)
			v.push_back(S_l1[i][f]);
		if (v.size() == 0)
		{
			cout << "emptyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy" << endl;
			continue;
		}
		S_l2[i] = outputLayer.execute_r(v)[0];
		v.clear();
		//cout<<S_l2[i] <<endl;
	}
	scores_arrays *sa = (scores_arrays*)malloc(8);
	sa->s1 = S_l1;
	sa->s2 = S_l2;
	//for (int i=0;i<X.size();i++)
	//	delete [] S_l1[i];
	//delete [] S_l1;
	//delete [] S_l2;
	//free(sa);
	return sa;

}

scores_arrays* kitsune::KitsuneTestTrain(vector<vector<double> > X, vector<int> mapIdx, vector<dA*> &ensLayer, dA &outputLayer, map < int, vector<int> > aeMap) {
	//######### INIT ##########
	//cout<<"init"<<endl;
	//# construct ensemble layer
	//aeIDs = np.unique(map)

	//#init output variables
	double ** S_l1 = new double*[X.size()];
	for (int i = 0; i<X.size(); i++)
		S_l1[i] = new double[ensLayer.size()];
	double *S_l2 = new double[X.size()];

	//########## TRAIN ##########
	//cout<<"train"<<endl;
	for (int i = 0; i<X.size(); i++)
	{
		//if (i % 1000 == 0)
		//  cout<<i<<endl;
		vector<double> x = X[i];
		//## EnsLayer
		for (int j = 0; j<ensLayer.size(); j++)
		{

			// #make sub inst
			vector<double> xi;
			for (int k = 0; k<aeMap[j].size(); k++)
				xi.push_back(x[aeMap[j][k]]);
			if (xi.size() == 0)
				continue;
			//x[aeMap[ensLayer[j]]];

			S_l1[i][j] = ensLayer[j]->train(xi)[0];


		}
		//## OutputLayer
		vector<double> v;
		//std::copy(v.begin(), v.end(), S_l1[i]);
		for (int r = 0; r < ensLayer.size(); r++)
			v.push_back(S_l1[0][r]);



		if (v.size() == 0)
			continue;
		S_l2[i] = outputLayer.train(v)[0];

	}


	scores_arrays *sa = (scores_arrays*)malloc(8);
	sa->s1 = S_l1;
	sa->s2 = S_l2;

	/*for (int i=0;i<X.size();i++)
	delete [] S_l1[i];
	delete [] S_l1;
	delete [] S_l2;
	free(sa);*/

	return sa;
}

vector<int> kitsune::convertMappingFileToVector(string path)
{
	std::ifstream infile(path);
	/* vector<int> v ;
	std::ifstream infile(path);
	int a;
	while (infile >> a)
	{
	cout<<a<<endl;
	v.push_back(a);
	}
	return v;*/

	std::vector<int> arr;
	int number;

	while (infile >> number) {
		arr.push_back(number);

		// ignore anything else on the line
		//file_handler.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	}
	return arr;
}

