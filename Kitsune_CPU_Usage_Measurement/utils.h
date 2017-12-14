#pragma once


#include <iostream>
#include <math.h>
#include <vector>
#include <stdlib.h>
//#define RAND_MAX 1000
#define  D_SCL_SECURE_NO_WARNINGS

#define _CRT_SECURE_NO_WARNINGS

using namespace std;
namespace utils {

  static double uniform(double min, double max) {
    return rand() / (RAND_MAX + 1.0) * (max - min) + min;
  }

 static  int binomial(int n, double p) {
    if(p < 0 || p > 1) return 0;
  
    int c = 0;
    double r;
  
    for(int i=0; i<n; i++) {
      r = rand() / (RAND_MAX + 1.0);
      if (r < p) c++;
    }

    return c;
  }

 static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
  }

 static double invLogCDF(double x, double loc=0, double scale=1)
 {
     return log(erfc((loc - x) / (scale * sqrt(2))) / 2);
 }

 static vector<string> splitString(string str,string delimiter){

     vector<string> tokens;
          size_t prev = 0, pos = 0;
          do
          {
              pos = str.find(delimiter, prev);
              if (pos == string::npos) pos = str.length();
              string token = str.substr(prev, pos-prev);
              if (!token.empty())
                  tokens.push_back(token);
              prev = pos + delimiter.length();
          }
          while (pos < str.length() && prev < str.length());
          return tokens;}

}

static double RMSE(const vector<double> &vec)
{
    double rmse = 0;
    for(int i = 0; i< vec.size();i++)
        rmse += pow(vec[i],2);
    rmse = sqrt(rmse/vec.size());
    return rmse;
}
