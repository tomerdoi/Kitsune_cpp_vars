#define  D_SCL_SECURE_NO_WARNINGS


#define _CRT_SECURE_NO_WARNINGS
#ifndef DA_H
#define DA_H

#include "utils.h"
#include <iostream>
//#include "LSS/instance.h"
#include "da_params.h"
//#include "netStat/incStat.h"
#include <cmath>
#include "vector"
using namespace utils;
using namespace std;

class dA  {

public:

    dA_params params;
    vector<double> norm_max;
    vector<double> norm_min;
    int n;
    //incStat* rollmean;

    double **W;
    double *hbias;
    double *vbias;

    dA(){}
    dA(dA_params params);
    //dA(Json::Value jsonObj);
    ~dA();
    void get_corrupted_input(double*, double *);
    void get_hidden_values(double*,double*);
    void reconstruct(double*,double*);
    void get_reconstructed_input(double*,double*);
    int getNumOutputs();

    vector<double> train(vector<double> &x);
    // void train_batch(double **x,int batchSize, int n_iters=10);
    //double execute(ml_inst* inst,double t=-1);
    vector<double> execute_r(vector<double> &x);
    vector<double> execute_batch(double ** x,int batchSize,double * ts);
    // double ** reconstruct_batch(double **x,int batchSize);
    //double normal_logcdf(double, double, double);
    bool inGrace();



    //double Score(double *);
    //void fromJSON(Json::Value jsonObj);
    //Json::Value toJSON();

    dA* clone();//returns deep copy of self
};

class inclusive_dA
{
public:
    inclusive_dA(vector<double> Lambdas,double trainTime = 35*60); //lambdas of each FE, train duration in sec
    ~inclusive_dA();
    //vector<incStat> FEs; //for feature extraction
    double thr;
    //void process(ml_inst* inst); //returns raw_scores (regarless if in training or not)
    double trainTerminationTime; //unix time of when training will termintate

    void addTrainTime(double duration_sec);
    void setTrainTime(double duration_sec);
    bool inTraining(double curTime);

private:
    dA* AE;

};

#endif //DA_H
