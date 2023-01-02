#include "flsa_dp.hpp"
#include "clm.hpp"
#include "utils.hpp"
#include "predict.hpp"
#include "train_tvaclm.hpp"
#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>
#include <math.h>
#include <string>
#include <fstream>
#include <stdexcept>

using namespace std;

int main(){
    const double pi = 0.5;
    double M = 30;

    // string datalist[] = { "wlb", "winequality-white","winequality"};
    // string datalist[] = {"ESL","LEV","SWD","automobile","balance-scale","bondrate","car","contact-lenses","eucalyptus","newthyroid","pasture","squash-stored","squash-unstored","tae","toy","ERA","winequality-red"};
    // string datalist[] = {"abalone", "bank1-5","bank2-5","calhousing-5","census1-5","census2-5","computer1-5","computer2-5","housing","machine","pyrim","stock"};
    // string datalist[] = {"bank1-10","bank2-10","calhousing-10","census1-10","census2-10","computer1-10","computer2-10","housing10","machine10","pyrim10","stock10"};
    string datalist[] = {"bondrate"};
    string datadir_name = "ordinal-regression";
    for (auto dataname: datalist){
        std::ofstream myFile("./../tvaclm_exp/check_f/"+dataname+".csv");
        int datanum = 0;
        auto datanum_str = std::to_string(datanum);
        std::string filename = "/home/iyori/work/gam/ordinal_regression/orca/datasets2/" + datadir_name + "/"+dataname+"/matlab/train_"+ dataname+"." + datanum_str;
        std::string test_filename = "/home/iyori/work/gam/ordinal_regression/orca/datasets2/" + datadir_name + "/"+dataname+"/matlab/test_"+ dataname+"." + datanum_str;
        double best_lam = 0.1;

        // std::string filename = "/home/iyori/work/gam/ordinal_regression/orca/datasets2/bigdata/"+dataname+"/matlab/train_"+ dataname+"." + datanum_str;
        // std::string test_filename = "/home/iyori/work/gam/ordinal_regression/orca/datasets2/bigdata/"+dataname+"/matlab/test_"+ dataname+"." + datanum_str;


        std::cout << "dataname: " << dataname << " |  data number: " << datanum_str << "\n";


        vector<vector<double>> train_data = csv2vector(filename, 0);


        int n = train_data.size();
        if (n==0) {
            std::cout << "no file found " << endl;
            continue;
        }
        int yd = train_data[0].size() - 1;
        int d =  1;
        int ds = 0;
        vector<vector<double>> x(d,vector<double>{});
        vector<int> y{};
        // TODO: remove
        for (int i=0; i<n; i++){
            if (i%5!=0){
                for (int j=ds; j<ds+d; j++){
                    x[j-ds].push_back(train_data[i][j]);
                }
                y.push_back((int) train_data[i][yd]);
            }
        }
        cout <<endl;
        n = y.size();
        //end

        // read test data;
        vector<vector<double>> test_data = csv2vector(test_filename, 0);
        int test_n = test_data.size();
        vector<vector<double>> test_x(d,vector<double>{});
        vector<int> test_y{};
        for (int i=0; i<test_n; i++){
            for (int j=ds; j<ds+d; j++){
                test_x[j-ds].push_back(test_data[i][j]);
            }
            test_y.push_back((int) test_data[i][yd]);
        }
        int q = *max_element(y.begin(), y.end());


        std::cout << "test_n: " << test_n << ", train_n: " << n << ", q: " << q <<"\n";

        std::cout << "x max: [";
        for (int j=0; j<d; j++) std::cout << *std::max_element(x[j].begin(), x[j].end()) << ' ';
        std::cout << "]" <<endl;

        std::cout << "x min: [";
        for (int j=0; j<d; j++) std::cout << *std::min_element(x[j].begin(), x[j].end()) << ' ';
        std::cout << "]" <<endl;

        // CLM
        std::cout <<"_______________________________________________\n";
        std::cout << "CLM: " << endl;
        double b[q - 1];
        for (int i=0; i < q-1; i++){
            b[i] =  2 * pi * i - pi * (q - 2);
        }
        vector<double> w(d+1);
        trainClm(x, y, q, w ,b, 0.5,M);
        std::cout << "b: ";
        for (int l = 0; l<q-1; l++){
            std::cout << b[l] << " ";
        }
        std::cout << "\n";

        double trainprob, trainacc, trainmae, testprob, testacc, testmae;
        clm_calc_metrix(w, x, y, trainmae, trainacc, trainprob, q, b,M);
        clm_calc_metrix(w, test_x, test_y, testmae, testacc, testprob, q, b,M);
         std::cout<< "____________________________________________________\n";

        std::cout << "TRAIN PROB: " << trainprob <<"| "<< "TRAIN ACC: " << trainacc <<"| "<< "TRAIN MAE: " << trainmae <<endl;
        std::cout << "TEST PROB: " << testprob <<"| "<< "TEST ACC: " << testacc <<"| "<< "TEST MAE: " << testmae <<endl;
        std::cout<< "++++++++++++++++++++++++++++++++++++++++++++++++++++\n" << endl;



        std::cout <<"_______________________________________________\n\n";


        // TVACLM
        std::cout <<"TVACLM:\n";
        vector<vector<double>> sortedf(d, vector<double>(n));
        vector<vector<double>> sortedx(d, vector<double>(n));
        vector<vector<int>> argsort(d, vector<int>(n));
        vector<vector<int>> argsort_c(d, vector<int>(n-1));
        vector<vector<int>> argsort_inv(d, vector<int>(n));
        set_argsort(argsort, argsort_c, argsort_inv, x);
        vector<vector<double>> f(d, vector<double>(y.size()));

        std::cout << "lambda: " << best_lam << ", M: " <<M<<"\n";
        // double b[q - 1]
        for (int i=0; i < q-1; i++){
            b[i] =  2 * pi * i - pi * (q - 2);
        }

        cout << x[0].size() <<"__"<< q<<"__"<< best_lam<<"__"<< M << endl;

        for (auto &xi :x[0]) cout << xi << " ";
        cout << endl;

        for (auto &yi :y) cout << yi << " ";
        cout << endl;
        int iteration;
        int invalid =  train_tvaclm(x, f, y, q, best_lam, b, 0.3, M,iteration);
        if (invalid) std::cout << "error!!" << endl;

        std::cout << "b: ";
        for (int l = 0; l<q-1; l++){
            std::cout << b[l] << " ";
        }
        std::cout << "\n";

        for (int j = 0; j< d;j++){
            for (int k = 0; k < n; k++){
                sortedf[j][k] = f[j][argsort[j][k]];
                sortedx[j][k] = x[j][argsort[j][k]];
            }
        }

        tvaclm_calc_metrix(sortedf,sortedx, x, y, trainmae, trainacc, trainprob, q, b,  M);
        tvaclm_calc_metrix(sortedf,sortedx, test_x, test_y, testmae, testacc, testprob, q, b,  M);
        
        // Send the column name to the stream
        
        // Send data to the stream
        for (int j = 0; j < d; j++){
            for (int i = 0; i < n; i++){
                myFile << f[j][i];
                if (i!=n-1) myFile << ",";
                else myFile << "\n";
            }
            
        }
        std::cout<< "____________________________________________________\n";

        std::cout << "TRAIN PROB: " << trainprob <<"| "<< "TRAIN ACC: " << trainacc <<"| "<< "TRAIN MAE: " << trainmae <<endl;
        std::cout << "TEST PROB: " << testprob <<"| "<< "TEST ACC: " << testacc <<"| "<< "TEST MAE: " << testmae <<endl;
        std::cout<< "++++++++++++++++++++++++++++++++++++++++++++++++++++\n" << endl;

        
        // Close the file
        myFile.close();
    }

    std::cout<< "==========================================================\n\n" << endl;

}