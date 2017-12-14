#define  D_SCL_SECURE_NO_WARNINGS

#define _CRT_SECURE_NO_WARNINGS

#include <cstdio>
#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

#include <vector>
#include "modelex.h"
#include "kitsune.h"
#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
void getMeasurements (const char * pathX, string pathM,string name, int maxClusterSize)
{
    vector<int> v;
    vector< vector <double> > X;
    kitsune k;
    modelEx m;
    double time,totalTime=0;
    std::ifstream infile(pathX);
    cout<<"get mapping"<<endl;
     v=k.convertMappingFileToVector(pathM);
	cout<<"Mapping is "<<endl;
	for (int h=0;h<v.size();h++)
	cout<<v[h]<<"_";
	cout<<endl;
     cout<<"loading dataset"<<endl;
     int thresh = 0;
	

	 ofstream myfile_scores;
	 ofstream myfile_CPU_Usage;
	ofstream myfile_times;
	//const char * p="scores_"+name+""+maxClusterSize+".txt";


	//p2
	char *p2 =(char *)malloc(200);
	strcpy(p2,"scores_");
	cout<<"got here "<<p2<<" "<<endl;
	strcat(p2,name.c_str());
	
	string part;
	part="_";

	strcat(p2,part.c_str());
	std::string s=std::to_string(maxClusterSize);
	
	strcat(p2,s.c_str());
	part=".txt";
	strcat(p2,part.c_str());


	//p3
	char *p3 = (char *)malloc(200);
	strcpy(p3, "times_");
	cout << "got here " << p3 << " " << endl;
	strcat(p3, name.c_str());

	part;
	part = "_";

	strcat(p3, part.c_str());
	 s = std::to_string(maxClusterSize);

	strcat(p3, s.c_str());
	part = ".txt";
	strcat(p3, part.c_str());
	//+name+"_"+maxClusterSize+".txt";
	//const char *p1=p2;*/
	

	//p4

	char *p4 = (char *)malloc(200);
	strcpy(p4, "CPU_Usage_");
	cout << "got here " << p4 << " " << endl;
	strcat(p4, name.c_str());

	part;
	part = "_";

	strcat(p4, part.c_str());
	s = std::to_string(maxClusterSize);

	strcat(p4, s.c_str());
	part = ".txt";
	strcat(p4, part.c_str());

	//end of P's
        myfile_scores.open (p2,std::ios_base::app);
		myfile_times.open(p3, std::ios_base::app);
		myfile_CPU_Usage.open(p3, std::ios_base::app);

         if (name == "ps2")
             thresh = 1000;
         else
             thresh = 1000000;
		 if (name != "ps2")
			 thresh -= 1000;
        //mapping build

         set<int> aeIDs;

         for (int i=0;i<v.size();i++)
              aeIDs.insert(v[i]);

        map < int , vector<int> > aeMap;
      //   for id in aeIDs:
      //       aeMap.append(np.where(map==id)[0])
          for (int i=0;i<v.size();i++)
              aeMap[v[i]].push_back(i);
         //architecture build
         vector<dA*> ensLayer;
	
         dA * d;
         for(int i=0;i<aeMap.size();i++)
         {
            dA_params *params=new dA_params (aeMap[i].size(),0,0.01,0,0,0.7);
            if (aeMap[i].size()==0)
                continue;
             d=new dA(*params);
             ensLayer.push_back(d);
         }

		 map < int, vector<int> > aeMapZipped;
		 int aeMapZippedIdx = 0;
		 for (int i = 0; i<aeMap.size(); i++)
		 {
			
			 if (aeMap[i].size() != 0)
			 {
				 aeMapZipped[aeMapZippedIdx] = aeMap[i];
				 aeMapZippedIdx++;
			 }
			
		 }
         //# construct output layer
         dA_params params (aeIDs.size(), 0, 0.01, 0, 0, 0.7);
         dA outputLayer (params);

         //threshold determination
		 high_resolution_clock::time_point t1, t2;
		 long double diff = 0;
		


         cout<<"Train"<<endl;
cout<<"The size of ensLayer when training is "<<ensLayer.size()<<endl;
	int i;
     for (i=0;i<thresh;i++)
     {
	if (i%1000==0)
		cout<<i<<endl;
         X=m.loadRow(infile);
		 
       // time=clock();
		 t1 = high_resolution_clock::now();
		 scores_arrays * scores = k.KitsuneTestTrain(X,v,ensLayer,outputLayer,aeMapZipped);
        //time=clock()-time;
		 t2 = high_resolution_clock::now();

		 long double score = scores->s2[0];

		 std::stringstream ss1;

		 ss1.precision(std::numeric_limits<long double>::digits10);//override the default
		 ss1 << score;
		 myfile_scores << ss1.str() << endl;


		  diff = duration_cast<nanoseconds>(t2 - t1).count();

		  std::stringstream ss;

		  ss.precision(std::numeric_limits<long double>::digits10);//override the default
		  ss << diff;

		  myfile_times << ss.str() << endl;
        //totalTime+=time;
	X[0].clear();
	X.clear();
     }

cout<<"The size of ensLayer when executing is "<<ensLayer.size()<<endl;
    cout<<"Execute"<<endl;
     while (!infile.eof())
     {
	if (i%1000==0)
		cout<<i<<endl;
		
         X=m.loadRow(infile);
		 if (X[0].size() == 1)
			 continue;
        //time=clock();
		 t1 = high_resolution_clock::now();

         scores_arrays * scores=k.KitsuneTestExecute(X,v,ensLayer,outputLayer, aeMapZipped);
		 t2 = high_resolution_clock::now();


		 diff = duration_cast<nanoseconds>(t2 - t1).count();

		 std::stringstream ss;

		 ss.precision(std::numeric_limits<long double>::digits10);//override the default
		 ss << diff;

		 myfile_times << ss.str() << endl;
		 for (int q = 0; q < X.size(); q++)
		 {
			 
			 long double score = scores->s2[q];

			 std::stringstream ss;

			 ss.precision(std::numeric_limits<long double>::digits10);//override the default
			 ss << score;
			 myfile_scores <<ss.str() << endl;
		 }
	//free memory
	for (int k=0;k<X.size();k++)
		delete [] scores->s1[k];
	delete scores->s1;
   delete [] scores->s2;
    free(scores);


        //time=clock()-time;
        //totalTime+=time;
	X[0].clear();
	X.clear();
	i++;

     }
	 myfile_scores.close();
	 myfile_times.close();
	 myfile_CPU_Usage.close();
    //writing runtime to file
      /*ofstream myfile;
	const char * p="runtime.txt"; 
        myfile.open (p,std::ios_base::app);
        myfile << "The time for "<<name<<"with max cluster size "<<i<<" is "<<totalTime<<endl;
        myfile.close();
 myfile_scores.close();
    cout<<"The time for "<<pathX<<" is "<<totalTime<<endl;*/
}

int main(int argc, char *argv[])
{
    //QCoreApplication a(argc, argv);
   //double d= (double) CLOCKS_PER_SEC;
   //cout<<"Clocks per sec is "<<d<<endl;
	int i;
	//string DSPaths[] = { "D:/datasets/KitsuneDatasets/ps2.csv","D:/datasets/ProfilloTDatasets/dalmini/WithBenign_gafgyt_attacks_combo.csv","D:/datasets/ProfilloTDatasets/dalmini/WithBenign_gafgyt_attacks_junk.csv", "D:/datasets/ProfilloTDatasets/dalmini/WithBenign_gafgyt_attacks_scan.csv","D:/datasets/ProfilloTDatasets/dalmini/WithBenign_gafgyt_attacks_tcp.csv","D:/datasets/ProfilloTDatasets/dalmini/WithBenign_gafgyt_attacks_udp.csv","D:/datasets/ProfilloTDatasets/dalmini/WithBenign_mirai_attacks_ack.csv","D:/datasets/ProfilloTDatasets/dalmini/WithBenign_mirai_attacks_scan.csv","D:/datasets/ProfilloTDatasets/dalmini/WithBenign_mirai_attacks_syn.csv", "D:/datasets/ProfilloTDatasets/dalmini/WithBenign_mirai_attacks_udp.csv","D:/datasets/ProfilloTDatasets/dalmini/WithBenign_mirai_attacks_udpplain.csv","D:/datasets/ProfilloTDatasets/ecobee/WithBenign_gafgyt_attacks_combo.csv","D:/datasets/ProfilloTDatasets/ecobee/WithBenign_gafgyt_attacks_junk.csv","D:/datasets/ProfilloTDatasets/ecobee/WithBenign_gafgyt_attacks_scan.csv","D:/datasets/ProfilloTDatasets/ecobee/WithBenign_gafgyt_attacks_tcp.csv","D:/datasets/ProfilloTDatasets/ecobee/WithBenign_gafgyt_attacks_udp.csv","D:/datasets/ProfilloTDatasets/ecobee/WithBenign_mirai_attacks_ack.csv","D:/datasets/ProfilloTDatasets/ecobee/WithBenign_mirai_attacks_scan.csv","D:/datasets/ProfilloTDatasets/ecobee/WithBenign_mirai_attacks_syn.csv","D:/datasets/ProfilloTDatasets/ecobee/WithBenign_mirai_attacks_udp.csv","D:/datasets/ProfilloTDatasets/ecobee/WithBenign_mirai_attacks_udpplain.csv","D:/datasets/ProfilloTDatasets/ennio/WithBenign_gafgyt_attacks_combo.csv","D:/datasets/ProfilloTDatasets/ennio/WithBenign_gafgyt_attacks_junk.csv","D:/datasets/ProfilloTDatasets/ennio/WithBenign_gafgyt_attacks_scan.csv","D:/datasets/ProfilloTDatasets/ennio/WithBenign_gafgyt_attacks_tcp.csv","D:/datasets/ProfilloTDatasets/ennio/WithBenign_gafgyt_attacks_udp.csv","D:/datasets/ProfilloTDatasets/phillips/WithBenign_gafgyt_attacks_combo.csv","D:/datasets/ProfilloTDatasets/phillips/WithBenign_gafgyt_attacks_junk.csv","D:/datasets/ProfilloTDatasets/phillips/WithBenign_gafgyt_attacks_scan.csv","D:/datasets/ProfilloTDatasets/phillips/WithBenign_gafgyt_attacks_tcp.csv","D:/datasets/ProfilloTDatasets/phillips/WithBenign_gafgyt_attacks_udp.csv","D:/datasets/ProfilloTDatasets/phillips/WithBenign_mirai_attacks_ack.csv","D:/datasets/ProfilloTDatasets/phillips/WithBenign_mirai_attacks_scan.csv","D:/datasets/ProfilloTDatasets/phillips/WithBenign_mirai_attacks_syn.csv","D:/datasets/ProfilloTDatasets/phillips/WithBenign_mirai_attacks_udp.csv","D:/datasets/ProfilloTDatasets/phillips/WithBenign_mirai_attacks_udpplain.csv","D:/datasets/ProfilloTDatasets/provision737/WithBenign_gafgyt_attacks_combo.csv","D:/datasets/ProfilloTDatasets/provision737/WithBenign_gafgyt_attacks_junk.csv","D:/datasets/ProfilloTDatasets/provision737/WithBenign_gafgyt_attacks_scan.csv","D:/datasets/ProfilloTDatasets/provision737/WithBenign_gafgyt_attacks_tcp.csv","D:/datasets/ProfilloTDatasets/provision737/WithBenign_gafgyt_attacks_udp.csv","D:/datasets/ProfilloTDatasets/provision737/WithBenign_mirai_attacks_ack.csv","D:/datasets/ProfilloTDatasets/provision737/WithBenign_mirai_attacks_scan.csv","D:/datasets/ProfilloTDatasets/provision737/WithBenign_mirai_attacks_syn.csv","D:/datasets/ProfilloTDatasets/provision737/WithBenign_mirai_attacks_udp.csv","D:/datasets/ProfilloTDatasets/provision737/WithBenign_mirai_attacks_udpplain.csv","D:/datasets/ProfilloTDatasets/provision838/WithBenign_gafgyt_attacks_combo.csv","D:/datasets/ProfilloTDatasets/provision838/WithBenign_gafgyt_attacks_junk.csv","D:/datasets/ProfilloTDatasets/provision838/WithBenign_gafgyt_attacks_scan.csv","D:/datasets/ProfilloTDatasets/provision838/WithBenign_gafgyt_attacks_tcp.csv","D:/datasets/ProfilloTDatasets/provision838/WithBenign_gafgyt_attacks_udp.csv","D:/datasets/ProfilloTDatasets/provision838/WithBenign_mirai_attacks_ack.csv","D:/datasets/ProfilloTDatasets/provision838/WithBenign_mirai_attacks_scan.csv","D:/datasets/ProfilloTDatasets/provision838/WithBenign_mirai_attacks_syn.csv","D:/datasets/ProfilloTDatasets/provision838/WithBenign_mirai_attacks_udp.csv","D:/datasets/ProfilloTDatasets/provision838/WithBenign_mirai_attacks_udpplain.csv","D:/datasets/ProfilloTDatasets/samsung/WithBenign_gafgyt_attacks_combo.csv","D:/datasets/ProfilloTDatasets/samsung/WithBenign_gafgyt_attacks_junk.csv","D:/datasets/ProfilloTDatasets/samsung/WithBenign_gafgyt_attacks_scan.csv","D:/datasets/ProfilloTDatasets/samsung/WithBenign_gafgyt_attacks_tcp.csv","D:/datasets/ProfilloTDatasets/samsung/WithBenign_gafgyt_attacks_udp.csv","D:/datasets/ProfilloTDatasets/simplehome1002/WithBenign_gafgyt_attacks_combo.csv","D:/datasets/ProfilloTDatasets/simplehome1002/WithBenign_gafgyt_attacks_junk.csv","D:/datasets/ProfilloTDatasets/simplehome1002/WithBenign_gafgyt_attacks_scan.csv","D:/datasets/ProfilloTDatasets/simplehome1002/WithBenign_gafgyt_attacks_tcp.csv","D:/datasets/ProfilloTDatasets/simplehome1002/WithBenign_gafgyt_attacks_udp.csv","D:/datasets/ProfilloTDatasets/simplehome1002/WithBenign_mirai_attacks_ack.csv","D:/datasets/ProfilloTDatasets/simplehome1002/WithBenign_mirai_attacks_scan.csv","D:/datasets/ProfilloTDatasets/simplehome1002/WithBenign_mirai_attacks_syn.csv","D:/datasets/ProfilloTDatasets/simplehome1002/WithBenign_mirai_attacks_udp.csv","D:/datasets/ProfilloTDatasets/simplehome1002/WithBenign_mirai_attacks_udpplain.csv","D:/datasets/ProfilloTDatasets/simplehome1003/WithBenign_gafgyt_attacks_combo.csv","D:/datasets/ProfilloTDatasets/simplehome1003/WithBenign_gafgyt_attacks_junk.csv","D:/datasets/ProfilloTDatasets/simplehome1003/WithBenign_gafgyt_attacks_scan.csv","D:/datasets/ProfilloTDatasets/simplehome1003/WithBenign_gafgyt_attacks_tcp.csv","D:/datasets/ProfilloTDatasets/simplehome1003/WithBenign_gafgyt_attacks_udp.csv","D:/datasets/ProfilloTDatasets/simplehome1003/WithBenign_mirai_attacks_ack.csv","D:/datasets/ProfilloTDatasets/simplehome1003/WithBenign_mirai_attacks_scan.csv","D:/datasets/ProfilloTDatasets/simplehome1003/WithBenign_mirai_attacks_syn.csv","D:/datasets/ProfilloTDatasets/simplehome1003/WithBenign_mirai_attacks_udp.csv","D:/datasets/ProfilloTDatasets/simplehome1003/WithBenign_mirai_attacks_udpplain.csv" };
	string DSNames[] = { "ps2","dalmini_WithBenign_gafgyt_attacks_combo","dalmini_WithBenign_gafgyt_attacks_junk", "dalmini_WithBenign_gafgyt_attacks_scan","dalmini_WithBenign_gafgyt_attacks_tcp","dalmini_WithBenign_gafgyt_attacks_udp","dalmini_WithBenign_mirai_attacks_ack","dalmini_WithBenign_mirai_attacks_scan","dalmini_WithBenign_mirai_attacks_syn", "dalmini_WithBenign_mirai_attacks_udp","dalmini_WithBenign_mirai_attacks_udpplain","ecobee_WithBenign_gafgyt_attacks_combo","ecobee_WithBenign_gafgyt_attacks_junk","ecobee_WithBenign_gafgyt_attacks_scan","ecobee_WithBenign_gafgyt_attacks_tcp","ecobee_WithBenign_gafgyt_attacks_udp","ecobee_WithBenign_mirai_attacks_ack","ecobee_WithBenign_mirai_attacks_scan","ecobee_WithBenign_mirai_attacks_syn","ecobee_WithBenign_mirai_attacks_udp","ecobee_WithBenign_mirai_attacks_udpplain","ennio_WithBenign_gafgyt_attacks_combo","ennio_WithBenign_gafgyt_attacks_junk","ennio_WithBenign_gafgyt_attacks_scan","ennio_WithBenign_gafgyt_attacks_tcp","ennio_WithBenign_gafgyt_attacks_udp","phillips_WithBenign_gafgyt_attacks_combo","phillips_WithBenign_gafgyt_attacks_junk","phillips_WithBenign_gafgyt_attacks_scan","phillips_WithBenign_gafgyt_attacks_tcp","phillips_WithBenign_gafgyt_attacks_udp","phillips_WithBenign_mirai_attacks_ack","phillips_WithBenign_mirai_attacks_scan","phillips_WithBenign_mirai_attacks_syn","phillips_WithBenign_mirai_attacks_udp","phillips_WithBenign_mirai_attacks_udpplain","provision737_WithBenign_gafgyt_attacks_combo","provision737_WithBenign_gafgyt_attacks_junk","provision737_WithBenign_gafgyt_attacks_scan","provision737_WithBenign_gafgyt_attacks_tcp","provision737_WithBenign_gafgyt_attacks_udp","provision737_WithBenign_mirai_attacks_ack","provision737_WithBenign_mirai_attacks_scan","provision737_WithBenign_mirai_attacks_syn","provision737_WithBenign_mirai_attacks_udp","provision737_WithBenign_mirai_attacks_udpplain","provision838_WithBenign_gafgyt_attacks_combo","provision838_WithBenign_gafgyt_attacks_junk","provision838_WithBenign_gafgyt_attacks_scan","provision838_WithBenign_gafgyt_attacks_tcp","provision838_WithBenign_gafgyt_attacks_udp","provision838_WithBenign_mirai_attacks_ack","provision838_WithBenign_mirai_attacks_scan","provision838_WithBenign_mirai_attacks_syn","provision838_WithBenign_mirai_attacks_udp","provision838_WithBenign_mirai_attacks_udpplain","samsung_WithBenign_gafgyt_attacks_combo","samsung_WithBenign_gafgyt_attacks_junk","samsung_WithBenign_gafgyt_attacks_scan","samsung_WithBenign_gafgyt_attacks_tcp","samsung_WithBenign_gafgyt_attacks_udp","simplehome1002_WithBenign_gafgyt_attacks_combo","simplehome1002_WithBenign_gafgyt_attacks_junk","simplehome1002_WithBenign_gafgyt_attacks_scan","simplehome1002_WithBenign_gafgyt_attacks_tcp","simplehome1002_WithBenign_gafgyt_attacks_udp","simplehome1002_WithBenign_mirai_attacks_ack","simplehome1002_WithBenign_mirai_attacks_scan","simplehome1002_WithBenign_mirai_attacks_syn","simplehome1002_WithBenign_mirai_attacks_udp","simplehome1002_WithBenign_mirai_attacks_udpplain","simplehome1003_WithBenign_gafgyt_attacks_combo","simplehome1003_WithBenign_gafgyt_attacks_junk","simplehome1003_WithBenign_gafgyt_attacks_scan","simplehome1003_WithBenign_gafgyt_attacks_tcp","simplehome1003_WithBenign_gafgyt_attacks_udp","simplehome1003_WithBenign_mirai_attacks_ack","simplehome1003_WithBenign_mirai_attacks_scan","simplehome1003_WithBenign_mirai_attacks_syn","simplehome1003_WithBenign_mirai_attacks_udp","simplehome1003_WithBenign_mirai_attacks_udpplain" };
	//string DSPathsForMapping[] = { "D:/datasets/KitsuneDatasets/ps2","D:/datasets/ProfilloTDatasets/dalmini/WithBenign_gafgyt_attacks_combo","D:/datasets/ProfilloTDatasets/dalmini/WithBenign_gafgyt_attacks_junk", "D:/datasets/ProfilloTDatasets/dalmini/WithBenign_gafgyt_attacks_scan","D:/datasets/ProfilloTDatasets/dalmini/WithBenign_gafgyt_attacks_tcp","D:/datasets/ProfilloTDatasets/dalmini/WithBenign_gafgyt_attacks_udp","D:/datasets/ProfilloTDatasets/dalmini/WithBenign_mirai_attacks_ack","D:/datasets/ProfilloTDatasets/dalmini/WithBenign_mirai_attacks_scan","D:/datasets/ProfilloTDatasets/dalmini/WithBenign_mirai_attacks_syn", "D:/datasets/ProfilloTDatasets/dalmini/WithBenign_mirai_attacks_udp","D:/datasets/ProfilloTDatasets/dalmini/WithBenign_mirai_attacks_udpplain","D:/datasets/ProfilloTDatasets/ecobee/WithBenign_gafgyt_attacks_combo","D:/datasets/ProfilloTDatasets/ecobee/WithBenign_gafgyt_attacks_junk","D:/datasets/ProfilloTDatasets/ecobee/WithBenign_gafgyt_attacks_scan","D:/datasets/ProfilloTDatasets/ecobee/WithBenign_gafgyt_attacks_tcp","D:/datasets/ProfilloTDatasets/ecobee/WithBenign_gafgyt_attacks_udp","D:/datasets/ProfilloTDatasets/ecobee/WithBenign_mirai_attacks_ack","D:/datasets/ProfilloTDatasets/ecobee/WithBenign_mirai_attacks_scan","D:/datasets/ProfilloTDatasets/ecobee/WithBenign_mirai_attacks_syn","D:/datasets/ProfilloTDatasets/ecobee/WithBenign_mirai_attacks_udp","D:/datasets/ProfilloTDatasets/ecobee/WithBenign_mirai_attacks_udpplain","D:/datasets/ProfilloTDatasets/ennio/WithBenign_gafgyt_attacks_combo","D:/datasets/ProfilloTDatasets/ennio/WithBenign_gafgyt_attacks_junk","D:/datasets/ProfilloTDatasets/ennio/WithBenign_gafgyt_attacks_scan","D:/datasets/ProfilloTDatasets/ennio/WithBenign_gafgyt_attacks_tcp","D:/datasets/ProfilloTDatasets/ennio/WithBenign_gafgyt_attacks_udp","D:/datasets/ProfilloTDatasets/phillips/WithBenign_gafgyt_attacks_combo","D:/datasets/ProfilloTDatasets/phillips/WithBenign_gafgyt_attacks_junk","D:/datasets/ProfilloTDatasets/phillips/WithBenign_gafgyt_attacks_scan","D:/datasets/ProfilloTDatasets/phillips/WithBenign_gafgyt_attacks_tcp","D:/datasets/ProfilloTDatasets/phillips/WithBenign_gafgyt_attacks_udp","D:/datasets/ProfilloTDatasets/phillips/WithBenign_mirai_attacks_ack","D:/datasets/ProfilloTDatasets/phillips/WithBenign_mirai_attacks_scan","D:/datasets/ProfilloTDatasets/phillips/WithBenign_mirai_attacks_syn","D:/datasets/ProfilloTDatasets/phillips/WithBenign_mirai_attacks_udp","D:/datasets/ProfilloTDatasets/phillips/WithBenign_mirai_attacks_udpplain","D:/datasets/ProfilloTDatasets/provision737/WithBenign_gafgyt_attacks_combo","D:/datasets/ProfilloTDatasets/provision737/WithBenign_gafgyt_attacks_junk","D:/datasets/ProfilloTDatasets/provision737/WithBenign_gafgyt_attacks_scan","D:/datasets/ProfilloTDatasets/provision737/WithBenign_gafgyt_attacks_tcp","D:/datasets/ProfilloTDatasets/provision737/WithBenign_gafgyt_attacks_udp","D:/datasets/ProfilloTDatasets/provision737/WithBenign_mirai_attacks_ack","D:/datasets/ProfilloTDatasets/provision737/WithBenign_mirai_attacks_scan","D:/datasets/ProfilloTDatasets/provision737/WithBenign_mirai_attacks_syn","D:/datasets/ProfilloTDatasets/provision737/WithBenign_mirai_attacks_udp","D:/datasets/ProfilloTDatasets/provision737/WithBenign_mirai_attacks_udpplain","D:/datasets/ProfilloTDatasets/provision838/WithBenign_gafgyt_attacks_combo","D:/datasets/ProfilloTDatasets/provision838/WithBenign_gafgyt_attacks_junk","D:/datasets/ProfilloTDatasets/provision838/WithBenign_gafgyt_attacks_scan","D:/datasets/ProfilloTDatasets/provision838/WithBenign_gafgyt_attacks_tcp","D:/datasets/ProfilloTDatasets/provision838/WithBenign_gafgyt_attacks_udp","D:/datasets/ProfilloTDatasets/provision838/WithBenign_mirai_attacks_ack","D:/datasets/ProfilloTDatasets/provision838/WithBenign_mirai_attacks_scan","D:/datasets/ProfilloTDatasets/provision838/WithBenign_mirai_attacks_syn","D:/datasets/ProfilloTDatasets/provision838/WithBenign_mirai_attacks_udp","D:/datasets/ProfilloTDatasets/provision838/WithBenign_mirai_attacks_udpplain","D:/datasets/ProfilloTDatasets/samsung/WithBenign_gafgyt_attacks_combo","D:/datasets/ProfilloTDatasets/samsung/WithBenign_gafgyt_attacks_junk","D:/datasets/ProfilloTDatasets/samsung/WithBenign_gafgyt_attacks_scan","D:/datasets/ProfilloTDatasets/samsung/WithBenign_gafgyt_attacks_tcp","D:/datasets/ProfilloTDatasets/samsung/WithBenign_gafgyt_attacks_udp","D:/datasets/ProfilloTDatasets/simplehome1002/WithBenign_gafgyt_attacks_combo","D:/datasets/ProfilloTDatasets/simplehome1002/WithBenign_gafgyt_attacks_junk","D:/datasets/ProfilloTDatasets/simplehome1002/WithBenign_gafgyt_attacks_scan","D:/datasets/ProfilloTDatasets/simplehome1002/WithBenign_gafgyt_attacks_tcp","D:/datasets/ProfilloTDatasets/simplehome1002/WithBenign_gafgyt_attacks_udp","D:/datasets/ProfilloTDatasets/simplehome1002/WithBenign_mirai_attacks_ack","D:/datasets/ProfilloTDatasets/simplehome1002/WithBenign_mirai_attacks_scan","D:/datasets/ProfilloTDatasets/simplehome1002/WithBenign_mirai_attacks_syn","D:/datasets/ProfilloTDatasets/simplehome1002/WithBenign_mirai_attacks_udp","D:/datasets/ProfilloTDatasets/simplehome1002/WithBenign_mirai_attacks_udpplain","D:/datasets/ProfilloTDatasets/simplehome1003/WithBenign_gafgyt_attacks_combo","D:/datasets/ProfilloTDatasets/simplehome1003/WithBenign_gafgyt_attacks_junk","D:/datasets/ProfilloTDatasets/simplehome1003/WithBenign_gafgyt_attacks_scan","D:/datasets/ProfilloTDatasets/simplehome1003/WithBenign_gafgyt_attacks_tcp","D:/datasets/ProfilloTDatasets/simplehome1003/WithBenign_gafgyt_attacks_udp","D:/datasets/ProfilloTDatasets/simplehome1003/WithBenign_mirai_attacks_ack","D:/datasets/ProfilloTDatasets/simplehome1003/WithBenign_mirai_attacks_scan","D:/datasets/ProfilloTDatasets/simplehome1003/WithBenign_mirai_attacks_syn","D:/datasets/ProfilloTDatasets/simplehome1003/WithBenign_mirai_attacks_udp","D:/datasets/ProfilloTDatasets/simplehome1003/WithBenign_mirai_attacks_udpplain" };
	
	string DSPaths[] = { "D:/datasets/KitsuneDatasets/ps2.csv","D:/datasets/ProfilloTDatasets/dalmini/mergedWithBenign_gafgyt_attacks_combo.csv","D:/datasets/ProfilloTDatasets/dalmini/mergedWithBenign_gafgyt_attacks_junk.csv", "D:/datasets/ProfilloTDatasets/dalmini/mergedWithBenign_gafgyt_attacks_scan.csv","D:/datasets/ProfilloTDatasets/dalmini/mergedWithBenign_gafgyt_attacks_tcp.csv","D:/datasets/ProfilloTDatasets/dalmini/mergedWithBenign_gafgyt_attacks_udp.csv","D:/datasets/ProfilloTDatasets/dalmini/mergedWithBenign_mirai_attacks_ack.csv","D:/datasets/ProfilloTDatasets/dalmini/mergedWithBenign_mirai_attacks_scan.csv","D:/datasets/ProfilloTDatasets/dalmini/mergedWithBenign_mirai_attacks_syn.csv", "D:/datasets/ProfilloTDatasets/dalmini/mergedWithBenign_mirai_attacks_udp.csv","D:/datasets/ProfilloTDatasets/dalmini/mergedWithBenign_mirai_attacks_udpplain.csv","D:/datasets/ProfilloTDatasets/ecobee/mergedWithBenign_gafgyt_attacks_combo.csv","D:/datasets/ProfilloTDatasets/ecobee/mergedWithBenign_gafgyt_attacks_junk.csv","D:/datasets/ProfilloTDatasets/ecobee/mergedWithBenign_gafgyt_attacks_scan.csv","D:/datasets/ProfilloTDatasets/ecobee/mergedWithBenign_gafgyt_attacks_tcp.csv","D:/datasets/ProfilloTDatasets/ecobee/mergedWithBenign_gafgyt_attacks_udp.csv","D:/datasets/ProfilloTDatasets/ecobee/mergedWithBenign_mirai_attacks_ack.csv","D:/datasets/ProfilloTDatasets/ecobee/mergedWithBenign_mirai_attacks_scan.csv","D:/datasets/ProfilloTDatasets/ecobee/mergedWithBenign_mirai_attacks_syn.csv","D:/datasets/ProfilloTDatasets/ecobee/mergedWithBenign_mirai_attacks_udp.csv","D:/datasets/ProfilloTDatasets/ecobee/mergedWithBenign_mirai_attacks_udpplain.csv","D:/datasets/ProfilloTDatasets/ennio/mergedWithBenign_gafgyt_attacks_combo.csv","D:/datasets/ProfilloTDatasets/ennio/mergedWithBenign_gafgyt_attacks_junk.csv","D:/datasets/ProfilloTDatasets/ennio/mergedWithBenign_gafgyt_attacks_scan.csv","D:/datasets/ProfilloTDatasets/ennio/mergedWithBenign_gafgyt_attacks_tcp.csv","D:/datasets/ProfilloTDatasets/ennio/mergedWithBenign_gafgyt_attacks_udp.csv","D:/datasets/ProfilloTDatasets/phillips/mergedWithBenign_gafgyt_attacks_combo.csv","D:/datasets/ProfilloTDatasets/phillips/mergedWithBenign_gafgyt_attacks_junk.csv","D:/datasets/ProfilloTDatasets/phillips/mergedWithBenign_gafgyt_attacks_scan.csv","D:/datasets/ProfilloTDatasets/phillips/mergedWithBenign_gafgyt_attacks_tcp.csv","D:/datasets/ProfilloTDatasets/phillips/mergedWithBenign_gafgyt_attacks_udp.csv","D:/datasets/ProfilloTDatasets/phillips/mergedWithBenign_mirai_attacks_ack.csv","D:/datasets/ProfilloTDatasets/phillips/mergedWithBenign_mirai_attacks_scan.csv","D:/datasets/ProfilloTDatasets/phillips/mergedWithBenign_mirai_attacks_syn.csv","D:/datasets/ProfilloTDatasets/phillips/mergedWithBenign_mirai_attacks_udp.csv","D:/datasets/ProfilloTDatasets/phillips/mergedWithBenign_mirai_attacks_udpplain.csv","D:/datasets/ProfilloTDatasets/provision737/mergedWithBenign_gafgyt_attacks_combo.csv","D:/datasets/ProfilloTDatasets/provision737/mergedWithBenign_gafgyt_attacks_junk.csv","D:/datasets/ProfilloTDatasets/provision737/mergedWithBenign_gafgyt_attacks_scan.csv","D:/datasets/ProfilloTDatasets/provision737/mergedWithBenign_gafgyt_attacks_tcp.csv","D:/datasets/ProfilloTDatasets/provision737/mergedWithBenign_gafgyt_attacks_udp.csv","D:/datasets/ProfilloTDatasets/provision737/mergedWithBenign_mirai_attacks_ack.csv","D:/datasets/ProfilloTDatasets/provision737/mergedWithBenign_mirai_attacks_scan.csv","D:/datasets/ProfilloTDatasets/provision737/mergedWithBenign_mirai_attacks_syn.csv","D:/datasets/ProfilloTDatasets/provision737/mergedWithBenign_mirai_attacks_udp.csv","D:/datasets/ProfilloTDatasets/provision737/mergedWithBenign_mirai_attacks_udpplain.csv","D:/datasets/ProfilloTDatasets/provision838/mergedWithBenign_gafgyt_attacks_combo.csv","D:/datasets/ProfilloTDatasets/provision838/mergedWithBenign_gafgyt_attacks_junk.csv","D:/datasets/ProfilloTDatasets/provision838/mergedWithBenign_gafgyt_attacks_scan.csv","D:/datasets/ProfilloTDatasets/provision838/mergedWithBenign_gafgyt_attacks_tcp.csv","D:/datasets/ProfilloTDatasets/provision838/mergedWithBenign_gafgyt_attacks_udp.csv","D:/datasets/ProfilloTDatasets/provision838/mergedWithBenign_mirai_attacks_ack.csv","D:/datasets/ProfilloTDatasets/provision838/mergedWithBenign_mirai_attacks_scan.csv","D:/datasets/ProfilloTDatasets/provision838/mergedWithBenign_mirai_attacks_syn.csv","D:/datasets/ProfilloTDatasets/provision838/mergedWithBenign_mirai_attacks_udp.csv","D:/datasets/ProfilloTDatasets/provision838/mergedWithBenign_mirai_attacks_udpplain.csv","D:/datasets/ProfilloTDatasets/samsung/mergedWithBenign_gafgyt_attacks_combo.csv","D:/datasets/ProfilloTDatasets/samsung/mergedWithBenign_gafgyt_attacks_junk.csv","D:/datasets/ProfilloTDatasets/samsung/mergedWithBenign_gafgyt_attacks_scan.csv","D:/datasets/ProfilloTDatasets/samsung/mergedWithBenign_gafgyt_attacks_tcp.csv","D:/datasets/ProfilloTDatasets/samsung/mergedWithBenign_gafgyt_attacks_udp.csv","D:/datasets/ProfilloTDatasets/simplehome1002/mergedWithBenign_gafgyt_attacks_combo.csv","D:/datasets/ProfilloTDatasets/simplehome1002/mergedWithBenign_gafgyt_attacks_junk.csv","D:/datasets/ProfilloTDatasets/simplehome1002/mergedWithBenign_gafgyt_attacks_scan.csv","D:/datasets/ProfilloTDatasets/simplehome1002/mergedWithBenign_gafgyt_attacks_tcp.csv","D:/datasets/ProfilloTDatasets/simplehome1002/mergedWithBenign_gafgyt_attacks_udp.csv","D:/datasets/ProfilloTDatasets/simplehome1002/mergedWithBenign_mirai_attacks_ack.csv","D:/datasets/ProfilloTDatasets/simplehome1002/mergedWithBenign_mirai_attacks_scan.csv","D:/datasets/ProfilloTDatasets/simplehome1002/mergedWithBenign_mirai_attacks_syn.csv","D:/datasets/ProfilloTDatasets/simplehome1002/mergedWithBenign_mirai_attacks_udp.csv","D:/datasets/ProfilloTDatasets/simplehome1002/mergedWithBenign_mirai_attacks_udpplain.csv","D:/datasets/ProfilloTDatasets/simplehome1003/mergedWithBenign_gafgyt_attacks_combo.csv","D:/datasets/ProfilloTDatasets/simplehome1003/mergedWithBenign_gafgyt_attacks_junk.csv","D:/datasets/ProfilloTDatasets/simplehome1003/mergedWithBenign_gafgyt_attacks_scan.csv","D:/datasets/ProfilloTDatasets/simplehome1003/mergedWithBenign_gafgyt_attacks_tcp.csv","D:/datasets/ProfilloTDatasets/simplehome1003/mergedWithBenign_gafgyt_attacks_udp.csv","D:/datasets/ProfilloTDatasets/simplehome1003/mergedWithBenign_mirai_attacks_ack.csv","D:/datasets/ProfilloTDatasets/simplehome1003/mergedWithBenign_mirai_attacks_scan.csv","D:/datasets/ProfilloTDatasets/simplehome1003/mergedWithBenign_mirai_attacks_syn.csv","D:/datasets/ProfilloTDatasets/simplehome1003/mergedWithBenign_mirai_attacks_udp.csv","D:/datasets/ProfilloTDatasets/simplehome1003/mergedWithBenign_mirai_attacks_udpplain.csv" };
	string DSPathsForMapping[] = { "D:/datasets/KitsuneDatasets/ps2","D:/datasets/ProfilloTDatasets/dalmini/mergedWithBenign_gafgyt_attacks_combo","D:/datasets/ProfilloTDatasets/dalmini/mergedWithBenign_gafgyt_attacks_junk", "D:/datasets/ProfilloTDatasets/dalmini/mergedWithBenign_gafgyt_attacks_scan","D:/datasets/ProfilloTDatasets/dalmini/mergedWithBenign_gafgyt_attacks_tcp","D:/datasets/ProfilloTDatasets/dalmini/mergedWithBenign_gafgyt_attacks_udp","D:/datasets/ProfilloTDatasets/dalmini/mergedWithBenign_mirai_attacks_ack","D:/datasets/ProfilloTDatasets/dalmini/mergedWithBenign_mirai_attacks_scan","D:/datasets/ProfilloTDatasets/dalmini/mergedWithBenign_mirai_attacks_syn", "D:/datasets/ProfilloTDatasets/dalmini/mergedWithBenign_mirai_attacks_udp","D:/datasets/ProfilloTDatasets/dalmini/mergedWithBenign_mirai_attacks_udpplain","D:/datasets/ProfilloTDatasets/ecobee/mergedWithBenign_gafgyt_attacks_combo","D:/datasets/ProfilloTDatasets/ecobee/mergedWithBenign_gafgyt_attacks_junk","D:/datasets/ProfilloTDatasets/ecobee/mergedWithBenign_gafgyt_attacks_scan","D:/datasets/ProfilloTDatasets/ecobee/mergedWithBenign_gafgyt_attacks_tcp","D:/datasets/ProfilloTDatasets/ecobee/mergedWithBenign_gafgyt_attacks_udp","D:/datasets/ProfilloTDatasets/ecobee/mergedWithBenign_mirai_attacks_ack","D:/datasets/ProfilloTDatasets/ecobee/mergedWithBenign_mirai_attacks_scan","D:/datasets/ProfilloTDatasets/ecobee/mergedWithBenign_mirai_attacks_syn","D:/datasets/ProfilloTDatasets/ecobee/mergedWithBenign_mirai_attacks_udp","D:/datasets/ProfilloTDatasets/ecobee/mergedWithBenign_mirai_attacks_udpplain","D:/datasets/ProfilloTDatasets/ennio/mergedWithBenign_gafgyt_attacks_combo","D:/datasets/ProfilloTDatasets/ennio/mergedWithBenign_gafgyt_attacks_junk","D:/datasets/ProfilloTDatasets/ennio/mergedWithBenign_gafgyt_attacks_scan","D:/datasets/ProfilloTDatasets/ennio/mergedWithBenign_gafgyt_attacks_tcp","D:/datasets/ProfilloTDatasets/ennio/mergedWithBenign_gafgyt_attacks_udp","D:/datasets/ProfilloTDatasets/phillips/mergedWithBenign_gafgyt_attacks_combo","D:/datasets/ProfilloTDatasets/phillips/mergedWithBenign_gafgyt_attacks_junk","D:/datasets/ProfilloTDatasets/phillips/mergedWithBenign_gafgyt_attacks_scan","D:/datasets/ProfilloTDatasets/phillips/mergedWithBenign_gafgyt_attacks_tcp","D:/datasets/ProfilloTDatasets/phillips/mergedWithBenign_gafgyt_attacks_udp","D:/datasets/ProfilloTDatasets/phillips/mergedWithBenign_mirai_attacks_ack","D:/datasets/ProfilloTDatasets/phillips/mergedWithBenign_mirai_attacks_scan","D:/datasets/ProfilloTDatasets/phillips/mergedWithBenign_mirai_attacks_syn","D:/datasets/ProfilloTDatasets/phillips/mergedWithBenign_mirai_attacks_udp","D:/datasets/ProfilloTDatasets/phillips/mergedWithBenign_mirai_attacks_udpplain","D:/datasets/ProfilloTDatasets/provision737/mergedWithBenign_gafgyt_attacks_combo","D:/datasets/ProfilloTDatasets/provision737/mergedWithBenign_gafgyt_attacks_junk","D:/datasets/ProfilloTDatasets/provision737/mergedWithBenign_gafgyt_attacks_scan","D:/datasets/ProfilloTDatasets/provision737/mergedWithBenign_gafgyt_attacks_tcp","D:/datasets/ProfilloTDatasets/provision737/mergedWithBenign_gafgyt_attacks_udp","D:/datasets/ProfilloTDatasets/provision737/mergedWithBenign_mirai_attacks_ack","D:/datasets/ProfilloTDatasets/provision737/mergedWithBenign_mirai_attacks_scan","D:/datasets/ProfilloTDatasets/provision737/mergedWithBenign_mirai_attacks_syn","D:/datasets/ProfilloTDatasets/provision737/mergedWithBenign_mirai_attacks_udp","D:/datasets/ProfilloTDatasets/provision737/mergedWithBenign_mirai_attacks_udpplain","D:/datasets/ProfilloTDatasets/provision838/mergedWithBenign_gafgyt_attacks_combo","D:/datasets/ProfilloTDatasets/provision838/mergedWithBenign_gafgyt_attacks_junk","D:/datasets/ProfilloTDatasets/provision838/mergedWithBenign_gafgyt_attacks_scan","D:/datasets/ProfilloTDatasets/provision838/mergedWithBenign_gafgyt_attacks_tcp","D:/datasets/ProfilloTDatasets/provision838/mergedWithBenign_gafgyt_attacks_udp","D:/datasets/ProfilloTDatasets/provision838/mergedWithBenign_mirai_attacks_ack","D:/datasets/ProfilloTDatasets/provision838/mergedWithBenign_mirai_attacks_scan","D:/datasets/ProfilloTDatasets/provision838/mergedWithBenign_mirai_attacks_syn","D:/datasets/ProfilloTDatasets/provision838/mergedWithBenign_mirai_attacks_udp","D:/datasets/ProfilloTDatasets/provision838/mergedWithBenign_mirai_attacks_udpplain","D:/datasets/ProfilloTDatasets/samsung/mergedWithBenign_gafgyt_attacks_combo","D:/datasets/ProfilloTDatasets/samsung/mergedWithBenign_gafgyt_attacks_junk","D:/datasets/ProfilloTDatasets/samsung/mergedWithBenign_gafgyt_attacks_scan","D:/datasets/ProfilloTDatasets/samsung/mergedWithBenign_gafgyt_attacks_tcp","D:/datasets/ProfilloTDatasets/samsung/mergedWithBenign_gafgyt_attacks_udp","D:/datasets/ProfilloTDatasets/simplehome1002/mergedWithBenign_gafgyt_attacks_combo","D:/datasets/ProfilloTDatasets/simplehome1002/mergedWithBenign_gafgyt_attacks_junk","D:/datasets/ProfilloTDatasets/simplehome1002/mergedWithBenign_gafgyt_attacks_scan","D:/datasets/ProfilloTDatasets/simplehome1002/mergedWithBenign_gafgyt_attacks_tcp","D:/datasets/ProfilloTDatasets/simplehome1002/mergedWithBenign_gafgyt_attacks_udp","D:/datasets/ProfilloTDatasets/simplehome1002/mergedWithBenign_mirai_attacks_ack","D:/datasets/ProfilloTDatasets/simplehome1002/mergedWithBenign_mirai_attacks_scan","D:/datasets/ProfilloTDatasets/simplehome1002/mergedWithBenign_mirai_attacks_syn","D:/datasets/ProfilloTDatasets/simplehome1002/mergedWithBenign_mirai_attacks_udp","D:/datasets/ProfilloTDatasets/simplehome1002/mergedWithBenign_mirai_attacks_udpplain","D:/datasets/ProfilloTDatasets/simplehome1003/mergedWithBenign_gafgyt_attacks_combo","D:/datasets/ProfilloTDatasets/simplehome1003/mergedWithBenign_gafgyt_attacks_junk","D:/datasets/ProfilloTDatasets/simplehome1003/mergedWithBenign_gafgyt_attacks_scan","D:/datasets/ProfilloTDatasets/simplehome1003/mergedWithBenign_gafgyt_attacks_tcp","D:/datasets/ProfilloTDatasets/simplehome1003/mergedWithBenign_gafgyt_attacks_udp","D:/datasets/ProfilloTDatasets/simplehome1003/mergedWithBenign_mirai_attacks_ack","D:/datasets/ProfilloTDatasets/simplehome1003/mergedWithBenign_mirai_attacks_scan","D:/datasets/ProfilloTDatasets/simplehome1003/mergedWithBenign_mirai_attacks_syn","D:/datasets/ProfilloTDatasets/simplehome1003/mergedWithBenign_mirai_attacks_udp","D:/datasets/ProfilloTDatasets/simplehome1003/mergedWithBenign_mirai_attacks_udpplain" };


	//for (int j = 0; j <= 10; j++)
	//{
		//if (DSNames[j] == "ps2")
			//continue;
	//pre-check for ps2:
	/*const char * pathXForMapping1 = "D:/datasets/KitsuneDatasets/ps2";
	const char * pathX1 = "D:/datasets/KitsuneDatasets/ps2.csv";
	string name1 = "ps2";
	std::stringstream ss1;
	ss1 << pathXForMapping1 << "_mapping.csv";
	getMeasurements(pathX1, ss1.str(), name1, 30);*/

	
	const char * pathXForMapping = "D:/datasets/KitsuneDatasets/rtsp4_maps/RTSP_4-003";
	const char * pathX = "D:/datasets/KitsuneDatasets/RTSP_4-003.csv";
	string name = "RTSP4";
		for (i = 50 ;i <= 50; i++)
		{
			//std::stringstream ss;
			//ss<<"/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/ps2_mappingmaxCluster_"<<i<<"_.csv";
			//getMeasurements("/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/ps2.csv",ss.str(),"ps2",i);
			/*    getMeasurements("/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/SSDP_lab_1-002.csv","/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/SSDP_lab_1-002_mapping.csv","SSDP");
				getMeasurements("//media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/Passive_Sniffing_3-005.csv","/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/Passive_Sniffing_3-005_mapping.csv","PassiveSniffing");
				getMeasurements("/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/fuzzing.csv","/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/fuzzing_mapping.csv","fuzzing");
				getMeasurements("/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/port_scan.csv","/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/port_scan_mapping.csv","port_scan");
				getMeasurements("/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/RTSP.csv","/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/RTSP_mapping.csv","RTSP");
				getMeasurements("/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/ssl_renego.csv","/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/ssl_renego_mapping.csv","ssl_ren");
				getMeasurements("/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/SYN_lab_1-001.csv","/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/SYN_lab_1-001_mapping.csv","syn"); */
			std::stringstream ss2;
			//ss2 << "/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/RTSP_4-003_mappingmaxCluster_" << i << "_.csv";
			
			ss2 << pathXForMapping <<"_mappingmaxCluster_" << i << "_.csv";
			//getMeasurements("/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/RTSP_4-003.csv", ss2.str(), "rtsp4", i);
			getMeasurements(pathX, ss2.str(),name, i);
			exit(0);
		}
	//}
/*getMeasurements("/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/etterArp.csv","/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/etterArp_mapping.csv","etterArp");
getMeasurements("/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/phiddle_09_08.csv","/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/phiddle_09_08_mapping.csv","phiddle");
getMeasurements("/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/SSL_lab_1-004.csv","/media/root/3804D9B004D970FC/AE_Ensemble_RuntimeTests/datasets/SSL_lab_1-004_mapping.csv","SSL_lab_1-004");*/
//SSDP_lab_1-002
    return 1;
}
