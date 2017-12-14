#ifndef TESTSCRIPT_KITSUNE_H
#define TESTSCRIPT_KITSUNE_H
#include <string>
#include <vector>
#include <ctime>
#include <iostream>
#include <map>
#include "modelex.h"
#include "kitsune.h"
#define  D_SCL_SECURE_NO_WARNINGS


#define _CRT_SECURE_NO_WARNINGS
using namespace std;
class testscript_Kitsune
{
public:
    testscript_Kitsune();
    void run( vector<int>  m,string mappingName);
};

#endif // TESTSCRIPT_KITSUNE_H
