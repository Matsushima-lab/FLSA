#include "flsa_dp.hpp"
#include "utils.hpp"
#include "predict.hpp"
#include "solve_block_coordinate_descent.hpp"
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
    const double pi = 1;
    const int CVNUM = 5;
    const vector<double> lam_list = {0.0625, 0.125,0.25,0.5,1,2,4,8,16,32,64,128};
    const int lamn = lam_list.size();
 
    string datalist[] = { "winequality-white","wlb", "winequality"};
    // string datalist[] = {"ERA","ESL","LEV","SWD","automobile","balance-scale","bondrate","car","contact-lenses","eucalyptus","newthyroid","pasture","squash-stored","squash-unstored","tae","toy","dwinequality-red"};
    // string datalist[] = {"ERA","ESL","LEV","SWD","automobile","balance-scale","bondrate","car","contact-lenses","eucalyptus","newthyroid","pasture","squash-stored","squash-unstored","tae","toy","dwinequality-red"};
    
    for (auto dataname: datalist){
        double eta = 1.;
        std::ofstream myFile("./../tvaclm_exp/bigdata/"+dataname+".csv", ios::app);
        myFile << "num, lambda, trainprob,trainacc,trainmae,prob,acc,mae" << endl;
        for (int datanum = 0; datanum<=30; datanum++){
            auto datanum_str = std::to_string(datanum);
            // std::string filename = "/home/iyori/work/gam/ordinal_regression/orca/datasets2/ordinal-regression/"+dataname+"/matlab/train_"+ dataname+"." + datanum_str;
            // std::string test_filename = "/home/iyori/work/gam/ordinal_regression/orca/datasets2/ordinal-regression/"+dataname+"/matlab/test_"+ dataname+"." + datanum_str;

            std::string filename = "/home/iyori/work/gam/ordinal_regression/orca/datasets2/bigdata/"+dataname+"/matlab/train_"+ dataname+"." + datanum_str;
            std::string test_filename = "/home/iyori/work/gam/ordinal_regression/orca/datasets2/bigdata/"+dataname+"/matlab/test_"+ dataname+"." + datanum_str;


            std::cout << "dataname: " << dataname << " |  data number: " << datanum_str << "\n";


            vector<vector<double>> train_data = csv2vector(filename, 0);


            int n = train_data.size();
            if (n==0) {
                std::cout << "no file found " << endl;
                continue;
            }
            int d = train_data[0].size() - 1;


            vector<vector<double>> x(d,vector<double>{});
            vector<int> y{};
            for (int i=0; i<n; i++){
                for (size_t j=0; j<d; j++){
                    x[j].push_back(train_data[i][j]);
                }
                y.push_back((int) train_data[i][d]);
            }
            // read test data;
            vector<vector<double>> test_data = csv2vector(test_filename, 0);
            int test_n = test_data.size();
            vector<vector<double>> test_x(d,vector<double>{});
            vector<int> test_y{};
            for (int i=0; i<test_n; i++){
                for (size_t j=0; j<d; j++){
                    test_x[j].push_back(test_data[i][j]);
                }
                test_y.push_back((int) test_data[i][d]);
            }

            int q = *max_element(y.begin(), y.end());

            double M = pi * q * eta;

            std::cout << "test_n: " << test_n << ", train_n: " << n << ", q: " << q <<"\n";

            
            vector<vector<double>> sortedf(d, vector<double>(n));
            vector<vector<double>> sortedx(d, vector<double>(n));
            vector<vector<int>> argsort(d, vector<int>(n));
            vector<vector<int>> argsort_c(d, vector<int>(n-1));
            vector<vector<int>> argsort_inv(d, vector<int>(n));
            set_argsort(argsort, argsort_c, argsort_inv, x);
            vector<double> maesum(lamn, 0);

            std::cout << "progress:" << flush;
            for (int k = 0; k < CVNUM; k++){
                // std::cout << "lambda: " << lam << ", cv k:" << k << "\n";
                
                std::cout << " -" << flush;
                vector<vector<double>> train_x(d,vector<double>{});
                vector<vector<double>> val_x(d,vector<double>{});
                vector<int> train_y{};
                vector<int> val_y{};
                for (int i = 0; i < n; i++){
                    if (i%CVNUM==k){
                        for (int j = 0; j< d;j++){
                            val_x[j].push_back(x[j][i]);
                        }
                        val_y.push_back(y[i]);
                    } else {
                        for (int j = 0; j< d;j++){
                            train_x[j].push_back(x[j][i]);
                        }
                        train_y.push_back(y[i]);
                    }
                
                }
                int train_n = train_y.size();
                std::cout << "-" << flush;
                // std::cout << "train data sample number: " << train_n << " |  validation data sample number: " << val_y.size() << endl;
                vector<vector<int>> argsort_cv(d, vector<int>(train_n));
                vector<vector<int>> argsort_c_cv(d, vector<int>(train_n-1));
                vector<vector<int>> argsort_inv_cv(d, vector<int>(train_n));
                set_argsort(argsort_cv, argsort_c_cv, argsort_inv_cv, train_x);
                for (int l=0; l < lamn; l++){

                    const double lam = lam_list[l];
                    double b[q - 1];
                    for (int i=0; i < q-1; i++){
                        b[i] =  2 * pi * i - pi * (q - 2);
                    }
                    vector<vector<double>> fcv(d, vector<double>(train_y.size(),0));

                    int invalid = solve_block_coordinate_descent(train_x, fcv, train_y, q, lam, b, 0.3, M);

                    if (invalid) {
                        l -= 1;
                        eta += 0.2;
                        M = pi * q * eta;
                        std::cout << "init value of newton is not valid. lambda: " << lam << ", k: " << k << ", eta" << eta <<"\n";
                        continue;
                    }
                    vector<vector<double>> sortedfcv(d, vector<double>(train_n));
                    vector<vector<double>> sortedxcv(d, vector<double>(train_n));

                    for (int j = 0; j< d;j++){
                        for (int k = 0; k < train_n; k++){
                            sortedfcv[j][k] = fcv[j][argsort_cv[j][k]];
                            sortedxcv[j][k] = train_x[j][argsort_cv[j][k]];
                        }
                    }
                    // double prob, acc, mae;
                    // calc_metrix(sortedfcv,sortedxcv, train_x, train_y, mae, acc, prob, q, b,  M);
                    double valprob, valacc, valmae;
                    calc_metrix(sortedfcv,sortedxcv, val_x, val_y, valmae, valacc, valprob, q, b,  M);
                    maesum[l]+=valmae;

                    std::cout << ">" << flush;
                }
            }
            std::cout << endl;
            double min_mae_ave = 1e10;
            double best_lam = -1;
            for (int l=0; l < lamn; l++){
                const double lam = lam_list[l];
                double mae_ave = maesum[l]/CVNUM;
                std::cout << " |     lambda: " << lam << ", mean MAE: " << mae_ave << "       |\n";
                if (mae_ave < min_mae_ave) {
                    min_mae_ave = mae_ave;
                    best_lam = lam;
                }
            }
            if (best_lam < 0) std::cout << "ERROR: LAMBDA is wrong value";

            vector<vector<double>> f(d, vector<double>(y.size()));

            std::cout << "lambda: " << best_lam << ", eta: " <<eta<<"\n";
            double b[q - 1];
            for (int i=0; i < q-1; i++){
                b[i] =  2 * pi * i - pi * (q - 2);
            }
            solve_block_coordinate_descent(x, f, y, q, best_lam, b, 0.3, M);

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
            double trainprob, trainacc, trainmae, testprob, testacc, testmae;

            calc_metrix(sortedf,sortedx, x, y, trainmae, trainacc, trainprob, q, b,  M);
            calc_metrix(sortedf,sortedx, test_x, test_y, testmae, testacc, testprob, q, b,  M);
            
            // Send the column name to the stream
            
            // Send data to the stream
            myFile <<datanum<<","<< best_lam << ","<<trainprob << ","<< trainacc << "," << trainmae<< "," <<testprob<< "," <<testacc<< "," <<testmae << endl;
            std::cout<< "____________________________________________________\n";

            std::cout << "TRAIN PROB: " << trainprob <<"| "<< "TRAIN ACC: " << trainacc <<"| "<< "TRAIN MAE: " << trainmae <<endl;
            std::cout << "TEST PROB: " << testprob <<"| "<< "TEST ACC: " << testacc <<"| "<< "TEST MAE: " << testmae <<endl;

            std::cout<< "++++++++++++++++++++++++++++++++++++++++++++++++++++\n" << endl;

        }
            
            // Close the file
        myFile.close();
    }

    std::cout<< "==========================================================\n\n" << endl;

}