#include "da_params.h"

dA_params::dA_params(unsigned int N_visible, unsigned int N_hidden,float Lr, float Corruption_level,unsigned int GracePeriod,
                     float Hidden_ratio,float Rollmean_L):hiddenRatio(Hidden_ratio), corruption_level(Corruption_level)

{
    this->n_visible = N_visible; //# num of units in visible (input) layer
    this->n_hidden = N_hidden; //# num of units in hidden layer
    if(n_hidden == 0){
      this->n_hidden = (unsigned int)(std::ceil(n_visible*hiddenRatio));
        if(this->n_hidden==0)
            this->n_hidden = 1;
    }
    this->lr = Lr;
    this->gracePeriod = GracePeriod;
    this->rollmean_L = Rollmean_L;
}

dA_params::dA_params(){}

dA_params* dA_params::clone()
{
//    dA_params* newobj = new dA_params();
//    newobj->n_visible = n_visible;
//    newobj->n_hidden = n_hidden;
//    newobj->lr = lr;
//    newobj->corruption_level = corruption_level;
//    newobj->gracePeriod = gracePeriod;
//    newobj->hiddenRatio = hiddenRatio;
//    newobj->rollmean_L = rollmean_L;
//    return newobj;
    return this;
}
