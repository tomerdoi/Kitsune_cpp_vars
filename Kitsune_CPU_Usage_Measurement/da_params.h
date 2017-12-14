#ifndef DA_PARAMS_H
#define DA_PARAMS_H
#include <cmath>
#define  D_SCL_SECURE_NO_WARNINGS

#define _CRT_SECURE_NO_WARNINGS

class dA_params
{
public:
    dA_params(unsigned int n_visible, unsigned int n_hidden=0,float lr=0.1, float corruption_level=0.0,unsigned int gracePeriod=1000,
              float hidden_ratio=0.8,float rollmean_L=1000000000);

    dA_params();
    unsigned int n_visible; //num of units in visible (input) layer
    unsigned int n_hidden; // num of units in hidden layer
    float lr;
    float corruption_level;
    unsigned int gracePeriod;
    float hiddenRatio;
    float rollmean_L;

    dA_params* clone();
};

#endif // DA_PARAMS_H
