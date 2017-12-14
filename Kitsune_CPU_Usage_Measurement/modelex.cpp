
#include <iostream>
#include <fstream>
#include <iterator>

#include <sstream>
#include <vector>
#include <string>
#include "modelex.h"
using namespace std;

static std::vector<std::string> getNextLineAndSplitIntoTokens(std::istream& str)
{
    std::vector<std::string>   result;
    std::string                line;
    std::getline(str,line);

    std::stringstream          lineStream(line);
    std::string                cell;

    while(std::getline(lineStream,cell, ','))
    {
        result.push_back(cell);
    }
    // This checks for a trailing comma with no data after it.
    if (!lineStream && cell.empty())
    {
        // If there was a trailing comma then add an empty element.
        result.push_back("");
    }
    return result;
}


vector<vector<double> > modelEx::loadDataset(const char *path)
{
    string l;
    vector<vector<double> > mat;
    int number_of_lines=0;
    std::ifstream myfile(path);

       while (std::getline(myfile, l))
           ++number_of_lines;

    std::ifstream infile (path);
    int counter=0;

     vector<string> line=getNextLineAndSplitIntoTokens(infile);

    while (counter<number_of_lines)
    {
        vector<string> line=getNextLineAndSplitIntoTokens(infile);
        counter++;
        if (counter%1000==0)
            cout<<counter<<endl;


        vector<double> input;
        for (int i=0;i<line.size();i++)
           {

            double f = 0.0;

            std::stringstream ss;


            ss << line[i];
            ss >> f;
            input.push_back(f);
            }

    mat.push_back(input);
    }

cout<<"finished"<<endl;
    return mat;


}

vector<vector<double> > modelEx::loadRow(ifstream & myfile)
{
    string l;
    vector<vector<double> > mat;
    int number_of_lines=0;
   // std::ifstream myfile(path);

       //while (std::getline(myfile, l))
         //  ++number_of_lines;


    int counter=0;

   //  vector<string> line=getNextLineAndSplitIntoTokens(myfile);

    //while (counter<number_of_lines)
    //{
        vector<string> line=getNextLineAndSplitIntoTokens(myfile);
        counter++;
        if (counter%1000==0)
            cout<<counter<<endl;


        vector<double> input;
        for (int i=0;i<line.size();i++)
           {

            double f = 0.0;

            std::stringstream ss;


            ss << line[i];
            ss >> f;
            input.push_back(f);
            }

    mat.push_back(input);
    //}

//cout<<"finished"<<endl;
    return mat;


}



