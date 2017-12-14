#include <iostream>
#include <math.h>

#include "da_params.h"
//#include "netStat/incStat.h"

#include "dA.h"
using namespace std;
//using namespace utils;


//dA::dA(Json::Value jsonObj){
//    this->fromJSON(jsonObj);
//}

dA::dA(dA_params Params) {//}  int size, int n_v, int n_h, double **w, double *hb, double *vb) {
	this->params = Params;

	W = new double*[this->params.n_hidden];
	for (int i = 0; i<this->params.n_hidden; i++) W[i] = new double[this->params.n_visible];
	double a = 1.0 / this->params.n_visible;

	for (int i = 0; i<this->params.n_hidden; i++) {
		for (int j = 0; j<this->params.n_visible; j++) {
			W[i][j] = uniform(-a, a);
		}
	}

	hbias = new double[this->params.n_hidden];
	for (int i = 0; i<this->params.n_hidden; i++)
		hbias[i] = 0;

	vbias = new double[this->params.n_visible];
	for (int i = 0; i<this->params.n_visible; i++)
		vbias[i] = 0;

	//# for 0-1 normlaization
	this->norm_max = vector<double>(this->params.n_visible); //numpy.ones((self.params.n_visible,)) * -numpy.Inf
	for (int i = 0; i<this->params.n_visible; i++)
		this->norm_max.at(i) = -1000000000;

	//self.norm_min = numpy.ones((self.params.n_visible,)) * numpy.Inf
	this->norm_min = vector<double>(this->params.n_visible); //numpy.ones((self.params.n_visible,)) * -numpy.Inf
	for (int i = 0; i<this->params.n_visible; i++)
		this->norm_min[i] = 1000000000;
	this->n = 0;

	//#for score smoothing
	//    if (this->params.rollmean_L != INFINITY)
	//        this->rollmean = new incStat(this->params.rollmean_L);
	//    else
	//        this->rollmean =NULL;
}

dA::~dA() {
	//    cout << "~da "<<endl;

	for (int i = 0; i<this->params.n_hidden; i++) delete[] W[i];
	delete[] W;
	delete[] hbias;
	delete[] vbias;
	//    if(this->rollmean)
	//        delete this->rollmean;
	//    //   if(this->params)
	//       delete params;
}

//Puts into tilde_x the final array. If params.corruption_level is not 0, then the caller is responsible to delte tilde_x
void dA::get_corrupted_input(double *x, double *tilde_x) {
	if (this->params.corruption_level > 0.0) {
		double p = 1 - this->params.corruption_level;
		for (int i = 0; i<this->params.n_visible; i++) {
			if (x[i] == 0) {
				tilde_x[i] = 0;
			}
			else {
				tilde_x[i] = binomial(1, p);
			}
		}
	}
}

// Encode
void dA::get_hidden_values(double *x, double *y) {
	for (int i = 0; i<this->params.n_hidden; i++) {
		y[i] = 0;
		for (int j = 0; j<this->params.n_visible; j++) {
			y[i] += W[i][j] * x[j];
		}
		y[i] += hbias[i];
		y[i] = sigmoid(y[i]);
	}
}

// Decode
void dA::get_reconstructed_input(double *y, double *z) {
	for (int i = 0; i<this->params.n_visible; i++) {
		z[i] = 0;
		for (int j = 0; j<this->params.n_hidden; j++) {
			z[i] += W[j][i] * y[j];
		}
		z[i] += vbias[i];
		z[i] = sigmoid(z[i]);
	}
}

int dA::getNumOutputs()
{
	return params.n_visible;
}

vector<double> dA::train(vector<double> &x) {//int *x, double lr, double corruption_level) {
	vector<double> errors = vector<double>(this->params.n_visible, 0);
	double* xd = x.data();
	double *tilde_x = new double[this->params.n_visible];
	double *x_n = new double[this->params.n_visible];
	double *y = new double[this->params.n_hidden];
	double *z = new double[this->params.n_visible];
	this->n++;
	double *L_vbias = new double[this->params.n_visible];
	double *L_hbias = new double[this->params.n_hidden];

	//Update normailization variables
	for (int i = 0; i<params.n_visible; i++)
	{
		if (this->norm_max[i]<xd[i])
			this->norm_max[i] = xd[i];
		if (this->norm_min[i]>xd[i])
			this->norm_min[i] = xd[i];
	}

	//Nomrlaize x
	for (int i = 0; i<params.n_visible; i++)
	{
		x_n[i] = (xd[i] - this->norm_min[i]) / (this->norm_max[i] - norm_min[i] + 0.0000000000000001);
	}
	for (int i = 0; i<this->params.n_visible; i++)
		tilde_x[i] = x_n[i];

	get_corrupted_input(x_n, tilde_x);
	get_hidden_values(tilde_x, y);
	get_reconstructed_input(y, z);

	// vbias
	for (int i = 0; i<this->params.n_visible; i++) {
		L_vbias[i] = x_n[i] - z[i];
		vbias[i] += this->params.lr * L_vbias[i];
	}

	// hbias
	for (int i = 0; i<this->params.n_hidden; i++) {
		L_hbias[i] = 0;
		for (int j = 0; j<this->params.n_visible; j++) {
			L_hbias[i] += W[i][j] * L_vbias[j];
		}
		L_hbias[i] *= y[i] * (1 - y[i]);

		hbias[i] += this->params.lr * L_hbias[i];
	}

	// W
	for (int i = 0; i<this->params.n_hidden; i++) {
		for (int j = 0; j<this->params.n_visible; j++) {
			W[i][j] += this->params.lr * (L_hbias[i] * tilde_x[j] + L_vbias[j] * y[i]);
		}
	}

	double rmse = 0;
	for (int i = 0; i<params.n_visible; i++)
	{
		errors[i] = z[i] - x_n[i];
		rmse += errors[i] * errors[i];
	}
	rmse = double(rmse / this->params.n_visible);
	rmse = pow(rmse, 0.5);
	errors[0] = rmse;

	delete[] L_hbias;
	delete[] L_vbias;
	delete[] z;
	delete[] y;
	delete[] tilde_x;
	delete[] x_n;

	return errors;
}

void dA::reconstruct(double *x, double *z) {
	double *y = new double[this->params.n_hidden];

	get_hidden_values(x, y);
	get_reconstructed_input(y, z);

	delete[] y;
}

//double dA::execute(ml_inst* inst,double t){
//    double* x = inst->numeric_feat.data();
//    // #returns MSE of the reconstruction of x
//    double MSE = 0.0;
//    if (this->n < this->params.gracePeriod)
//        return MSE;

//    double *x_n = new double[this->params.n_visible];
//    for (int i=0;i<params.n_visible;i++)
//    {
//        x_n[i]=(x[i]-this->norm_min[i])/(this->norm_max[i]-norm_min[i]+0.0000000000000001);
//    }

//    double *  z = new double[this->params.n_visible];
//    this->reconstruct(x_n,z);
//    for (int i=0;i<params.n_visible;i++)
//    {
//        MSE+=pow(z[i]-x_n[i],2);
//    }
//    MSE/=params.n_visible;
//    delete [] z;
//    delete [] x_n;

//    //#rollmean?
//    if (this->rollmean != NULL){
//        this->rollmean->insert(MSE,t);
//        MSE = this->rollmean->getMean();
//    }
//    return MSE;
//}


vector<double> dA::execute_r(vector<double> &x) {
	double* xd = x.data();
	vector<double> errors = vector<double>(this->params.n_visible, 0);
	if (this->inGrace())
		return errors;

	double *x_n = new double[this->params.n_visible];
	for (int i = 0; i<this->params.n_visible; i++)
	{
		//x_n[i]=(xd[i]-this->norm_min[i])/(this->norm_max[i]-this->norm_min[i]+0.0000000000000001);
		x_n[i] = (xd[i] - this->norm_min[i]) / (this->norm_max[i] - this->norm_min[i] + 1);
	}

	double *  z = new double[this->params.n_visible];
	this->reconstruct(x_n, z);
	for (int i = 0; i<this->params.n_visible; i++)
	{
		errors[i] = z[i] - x_n[i];
	}
	double rmse = 0;
	for (int i = 0; i<this->params.n_visible; i++)
	{
		rmse += errors[i] * errors[i];
	}
	rmse = double(rmse / this->params.n_visible);
	rmse = pow(rmse, 0.5);
	delete[] z;
	delete[] x_n;
	errors[0] = rmse;
	return errors;
}

vector<double> dA::execute_batch(double ** x, int batchSize, double * ts)
{
	vector<double> MSEs(batchSize, 0);
	// #returns MSE of the reconstruction of x
	if (this->n < this->params.gracePeriod) {
		return MSEs;
	}

	//# 0-1 normalize
	double **x_n = new double*[batchSize];
	for (int i = 0; i<batchSize; i++)
	{
		x_n[i] = new double[this->params.n_visible];
		for (int j = 0; j<params.n_visible; j++)
		{
			x_n[i][j] = (x[i][j] - this->norm_min[j]) / (this->norm_max[j] - norm_min[j] + 0.0000000000000001);
		}
	}

	//allocate recontruction matrix
	double** z = new double*[batchSize];
	for (int i = 0; i< batchSize; i++)
		z[i] = new double[this->params.n_visible];

	for (int i = 0; i<batchSize; i++)
		this->reconstruct(x_n[i], z[i]);

	for (int i = 0; i<batchSize; i++)
	{
		for (int j = 0; j<params.n_visible; j++)
		{
			MSEs[i] += pow(z[i][j] - x_n[i][j], 2);
		}
		MSEs[i] /= params.n_visible;
	}

	//deallocate recontruction matrix
	for (int i = 0; i< batchSize; i++)
		delete[] z[i];
	delete[] z;
	//deallocate norm matrix
	for (int i = 0; i< batchSize; i++)
		delete[] x_n[i];
	delete[] x_n;

	//#rollmean?
	//    if (this->rollmean != NULL)
	//    {
	//        if (ts==NULL)
	//        {
	//            ts=new double [params.n_visible];
	//            for (int j=0;j<params.n_visible;j++)
	//                ts[j]=0;
	//        }
	//        for (int i=0;i<batchSize;i++)
	//        {
	//            this->rollmean->insert(MSEs[i],ts[i]); //check insert
	//            MSEs[i]=this->rollmean->getMean(); //
	//        }

	//    }
	return MSEs;

}

bool dA::inGrace()
{
	return this->n < this->params.gracePeriod;
}


dA* dA::clone()
{
	dA* newObj = new dA();
	newObj->n = n;
	//    newObj->params = params.clone();
	newObj->params = this->params;

	newObj->W = new double*[this->params.n_hidden];
	for (int i = 0; i<this->params.n_hidden; i++)
		newObj->W[i] = new double[this->params.n_visible];

	for (int i = 0; i<this->params.n_hidden; i++) {
		for (int j = 0; j<this->params.n_visible; j++) {
			newObj->W[i][j] = this->W[i][j];
		}
	}

	newObj->hbias = new double[this->params.n_hidden];
	for (int i = 0; i<this->params.n_hidden; i++) newObj->hbias[i] = this->hbias[i];

	newObj->vbias = new double[this->params.n_visible];
	for (int i = 0; i<this->params.n_visible; i++) newObj->vbias[i] = this->vbias[i];

	for (int i = 0; i<norm_max.size(); i++) newObj->norm_max.push_back(this->norm_max[i]);
	for (int i = 0; i<norm_min.size(); i++) newObj->norm_min.push_back(this->norm_min[i]);
	//    if (this->rollmean != NULL)
	//        newObj->rollmean = this->rollmean->clone();
	//    else
	//        newObj->rollmean = NULL;

	return newObj;
}

//Json::Value dA::toJSON()
//{
//    Json::Value jsonObj;
//    jsonObj["n_visible"]=this->params.n_visible;
//    jsonObj["n_hidden"]=this->params.n_hidden;
//    jsonObj["n"]=this->n;
//    jsonObj["lr"]=this->params.lr;
//    jsonObj["corruption_level"]=this->params.corruption_level;
//    jsonObj["grapcePeriod"]=this->params.gracePeriod;
//    jsonObj["hiddenRatio"]=this->params.hiddenRatio;
//    jsonObj["rollmean_L"]=this->params.rollmean_L;

//    //    double wArray [params.n_visible*params.n_hidden];
//    //    int counter=0;
//    //    for (int i=0;i<this->params.n_visible;i++)
//    //    {
//    //        for (int j=0;j<this->params.n_hidden;j++)
//    //        {
//    //            wArray[counter]=this->W[j][i];
//    //            counter++;
//    //        }
//    //    }
//    //    for(double value : wArray ){
//    //        jsonObj["W"].append(value);
//    //    }

//    for (int i=0;i<this->params.n_visible;i++)
//        for (int j=0;j<this->params.n_hidden;j++)
//            jsonObj["W"].append(this->W[j][i]);
//    for (int i=0;i<this->params.n_visible;i++)
//        jsonObj["vbias"].append(this->vbias[i]);
//    for (int j=0;j<this->params.n_hidden;j++)
//        jsonObj["hbias"].append(this->hbias[j]);
//    for(std::vector<double>::iterator it = norm_max.begin(); it != norm_max.end(); it++)
//        jsonObj["norm_max"].append(*it);
//    for(std::vector<double>::iterator it = norm_max.begin(); it != norm_max.end(); it++)
//        jsonObj["norm_min"].append(*it);


//    return jsonObj;


//}

//void dA::fromJSON(Json::Value jsonObj)
//{
//    int counter=0;
//    vector<double> doubleWEIGHTSVec;
//    //da params
//    this->params.n_visible=std::stoi( jsonObj.get("n_visible","").toStyledString());
//    this->params.n_hidden=std::stoi( jsonObj.get("n_hidden","").toStyledString());
//    this->params.corruption_level=jsonObj["corruption_level"].asFloat();
//    this->params.lr=jsonObj["lr"].asFloat();
//    this->params.gracePeriod=jsonObj["gracePeriod"].asUInt();
//    this->params.hiddenRatio=jsonObj["hiddenRatio"].asFloat();
//    this->params.rollmean_L=jsonObj["rollmean_L"].asFloat();

//    this->n=std::stoi( jsonObj.get("n",0).toStyledString());


//    this->norm_max.clear();
//    this->norm_min.clear();
//    //norm_max
//    const Json::Value& norm_maxV=jsonObj["norm_max"];
//    //vector<double> norm_max=*(new vector<double>());
//    for (int i=0;i<params.n_visible;i++)
//    {
//        norm_max.push_back(norm_maxV[i].asDouble());
//    }
//    //norm_min
//    const Json::Value& norm_minV=jsonObj["norm_min"];
//    //vector<double> norm_min=*(new vector<double>());

//    for (int i=0;i<params.n_visible;i++)
//    {
//        //norm_min.push_back(norm_minV[i].asDouble());
//        norm_min.push_back(norm_minV[i].asDouble());
//    }
//    //this->norm_max=norm_max;
//    //this->norm_min=norm_min;

//    //L_hbias
//    this->hbias=new double[this->params.n_hidden];

//    //L_vbias
//    this->vbias=new double[this->params.n_visible];

//    const Json::Value& weightsJ=jsonObj["W"];

//    const Json::Value& hbiasJ=jsonObj["hbias"];
//    const Json::Value& vbiasJ=jsonObj["vbias"];

//    this->W = new double*[this->params.n_hidden];
//    for(int i=0; i<this->params.n_hidden; i++) this->W[i] = new double[this->params.n_visible];

//    for (int i=0;i<params.n_visible*params.n_hidden;i++)
//    {
//        doubleWEIGHTSVec.push_back(weightsJ[i].asDouble());
//    }



//    for(int i=0; i<this->params.n_hidden; i++) {
//        for(int j=0; j<this->params.n_visible; j++) {
//            W[i][j] = doubleWEIGHTSVec.at(counter);

//            counter++;
//        }
//    }

//    //hbias
//    counter=0;
//    for (int i=0;i<params.n_hidden;i++)
//    {
//        this->hbias[counter] = hbiasJ[i].asDouble();
//        counter++;
//    }

//    //vbias
//    counter=0;
//    for (int i=0;i<params.n_visible;i++)
//    {
//        this->vbias[counter] = vbiasJ[i].asDouble();
//        counter++;
//    }
//    this->rollmean = new incStat(this->params.rollmean_L);

//}

//inclusive_dA::inclusive_dA(vector<double> Lambdas,double trainTime)
//{
//    thr = -INFINITY;
//    AE = new dA(dA_params(3,2,0.1,0,1000));
//    FEs = vector<incStat>(Lambdas.size());
//    for(int l = 0; l < Lambdas.size() ; l++)
//        FEs[l] = incStat(Lambdas[l]);
//    setTrainTime(trainTime);
//}

inclusive_dA::~inclusive_dA()
{
	delete AE;
}

//void inclusive_dA::process(ml_inst *inst)
//{
//    if(inTraining(inst->timestamp))
//    {
//        //train
//        inst->pred.input = inst->numeric_feat;
//        inst->pred.output = AE->train(inst->pred.input);
//        double raw_score = RMSE(inst->pred.output);
//        //Update Threshold
//        if(raw_score > thr)
//            thr = raw_score;
//        //compute anom score
//        inst->pred.score = raw_score/(5*thr);
//    }
//    else
//    {
//        //execute
//        inst->pred.input = inst->numeric_feat;
//        inst->pred.output = AE->execute_r(inst->pred.input);
//        double raw_score = RMSE(inst->pred.output);
//        //compute anom score
//        inst->pred.score = raw_score/(5*thr);
//    }
//}

/*void inclusive_dA::addTrainTime(double duration_sec) //duration is in seconds
{
if(duration_sec<=0)
return;
/* Get current time
struct timespec ct;
clock_gettime(CLOCK_REALTIME, &ct);
double curTime =  (double)ct.tv_sec + ((double)ct.tv_nsec)/1000000000;
if(inTraining(curTime))
trainTerminationTime += duration_sec;
else
trainTerminationTime = curTime + duration_sec;
}*/

bool inclusive_dA::inTraining(double curTime)
{
	if (curTime < trainTerminationTime)
		return true;
	else
		return false;
}
/*void inclusive_dA::setTrainTime(double duration_sec)
{
/* Setup train time
struct timespec curTime;
clock_gettime(CLOCK_REALTIME, &curTime); //get current time
trainTerminationTime =  (double)curTime.tv_sec + ((double)curTime.tv_nsec)/1000000000;
trainTerminationTime += duration_sec;
}*/
