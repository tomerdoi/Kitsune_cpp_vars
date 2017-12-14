#include "testscript_kitsune.h"

testscript_Kitsune::testscript_Kitsune()
{

}

void testscript_Kitsune::run( vector<int>  m,string mappingName){
    string RTSP = "D:\datasets\KitsuneDatasets\\RTSP.csv";
    string SSDP = "D:\datasets\KitsuneDatasets\SSDP_lab_1-002.csv";
    string SYN = "D:\datasets\KitsuneDatasets\SYN_lab_1-001.csv";
    string SSL = "D:\datasets\KitsuneDatasets\SSL_lab_1-004.csv";
    string PHD = "D:\datasets\KitsuneDatasets\Passive_Sniffing_3-005.csv";
    string PortS = "D:\datasets\KitsuneDatasets\port_scan.csv";
    string RTSP4 = "D:\datasets\KitsuneDatasets\RTSP_4-003.csv";
    string Fuzz = "D:\datasets\KitsuneDatasets\\fuzzing.csv";
    string PHD2 = "D:\datasets\KitsuneDatasets\\phiddle_09_08.csv";
    string ARP = "D:\datasets\KitsuneDatasets\\etterArp.csv";
    string SSL2 = "D:\datasets\KitsuneDatasets\\ssl_renego.csv";

    static string arr[] = {RTSP,SSDP,SYN,SSL,PHD,PortS,RTSP4,

                                Fuzz,PHD2,ARP,SSL2};
    modelEx me;

    int arrLength=11;
    string save_name_pref = "KitsuneC";
    int trainEnd = 1000000;

    for (int i=0;i<arrLength;i++)
    {
        cout<<"Loading "<<arr[i]<<endl;
        //X = pd.read_csv(path,header=0).as_matrix()
        vector<vector<double> > mat=me.loadDataset(arr[i].c_str());
        cout<<"Finding Map"<<endl;
        //start = time.time()
        double start=clock();
        //map = k_clust(np.transpose(X[1:trainEnd, ]), 10)
        double stop = clock();
        cout<<(stop - start)<<endl;

        // Run Kitsune
        cout<<"Running Kitsune:"<<endl;
        start = clock();
        kitsune kit;
       scores_arrays * OUT = kit.KitsuneTest(mat, m, trainEnd);
        stop = clock();
        cout<<(stop - start)<<endl;

        // Save Results
        cout<<"Saving Results..."<<endl;
        //string filename = str.split(path.split(sep="\\")[len(path.split(sep="\\"))-1],sep=".")[0];
        string filename=mappingName+arr[i].substr(arr[i].find_last_of("\\"));
         ofstream myfile (filename.c_str());
         for(int count = 0; count < mat.size(); count ++){
                myfile << OUT->s2[count] << "\n" ;
        //np.savetxt("RES_"+save_name_pref+"_"+filename+".csv",OUT[1],delimiter=',')
    }
    cout<<"done."<<endl;
}
}
