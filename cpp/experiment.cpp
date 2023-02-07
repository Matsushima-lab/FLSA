#include "flsa_dp.hpp"
#include "utils.hpp"
#include "predict.hpp"
#include "train_tvaclm.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <math.h>
#include <string>
#include <fstream>
#include <stdexcept>

using namespace std;

int main(){
    const int CVNUM = 5;
    int local_cvnum;
    bool invalid_data = false;
    vector<double> lam_list(12);
    for (double i=0; i< 12; i++){
        lam_list[i] = pow(10,(i/4.)-1);
    }
    // vector<double> lam_list{5};
    const int lamn = lam_list.size();
    int iteration;
 
    string datalist[] = { "wlb", "winequality-white","winequality"};
    // string datalist[] = { "wlb"};
    // string datalist[] = {"ESL","LEV","SWD","automobile","balance-scale","bondrate","car","contact-lenses","eucalyptus","newthyroid","pasture","squash-stored","squash-unstored","tae","toy","ERA","winequality-red"};
    // string datalist[] = {"abalone", "bank1-5","bank2-5","calhousing-5","census1-5","census2-5","computer1-5","computer2-5","housing","machine","pyrim","stock"};
    // string datalist[] = {"abalone10", "bank1-10","bank2-10","calhousing-10","census1-10","census2-10","computer1-10","computer2-10","housing10","machine10","pyrim10","stock10"};

    // string datadir_name = "ordinal-regression";
    // string datadir_name = "discretized-regression/10bins";
    // string datadir_name = "scale";
    string datadir_name = "bigdata";

    for (auto dataname: datalist){
        double pi = 1e-4;
        double M = 30;
        double duration;
        std::ofstream myFile("./../tvaclm_exp4/" + datadir_name + "/"+dataname+ ".csv");
        myFile << "num,lambda,iteration,duration,trainprob,trainacc,trainmae,prob,acc,mae" << endl;
        for (int datanum = 0; datanum<=30; datanum++){
            auto datanum_str = std::to_string(datanum);
            std::string filename = "/home/iyori/work/gam/ordinal_regression/orca/datasets2/" + datadir_name + "/"+dataname+"/matlab/train_"+ dataname+"." + datanum_str;
            std::string test_filename = "/home/iyori/work/gam/ordinal_regression/orca/datasets2/" + datadir_name + "/"+dataname+"/matlab/test_"+ dataname+"." + datanum_str;

            // std::string filename = "/home/iyori/work/gam/ordinal_regression/orca/datasets2/bigdata/"+dataname+"/matlab/train_"+ dataname+"." + datanum_str;
            // std::string test_filename = "/home/iyori/work/gam/ordinal_regression/orca/datasets2/bigdata/"+dataname+"/matlab/test_"+ dataname+"." + datanum_str;


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


            std::cout << "test_n: " << test_n << ", train_n: " << n << ", q: " << q <<"\n";

            
            vector<vector<double>> sortedf(d, vector<double>(n));
            vector<vector<double>> sortedx(d, vector<double>(n));
            vector<vector<int>> argsort(d, vector<int>(n));
            vector<vector<int>> argsort_c(d, vector<int>(n-1));
            vector<vector<int>> argsort_inv(d, vector<int>(n));
            set_argsort(argsort, argsort_c, argsort_inv, x);
            vector<double> maesum(lamn, 0);
            vector<int> itersum(lamn, 0);

            std::cout << "progress:" << flush;
            local_cvnum = CVNUM;
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
                for (int qe=1; qe<=q; qe++){
                    if (!std::count(train_y.begin(), train_y.end(), qe)){
                        std::cout << "element not found... continue" << endl;
                        invalid_data = true;          
                        break;
                    }
                }
                if (invalid_data){
                    std::cout << "#";
                    local_cvnum -= 1;
                    invalid_data = false;
                    continue;
                }

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
                    int invalid = train_tvaclm(train_x, fcv, train_y, q, lam, b, 0.3, M,iteration);
                    if (invalid) {
                        for (int i=0; i < q-1; i++) cout << b[i] << " ";
                        cout << endl;
                        cout << q <<" error\n";
                        return 1;
                        // l -= 1;
                        // M *= 2;
                        // pi /= 2;
                        // for (int bq=0; bq < q-1; bq++) cout << b[bq] << " ";
                        // cout << endl;
                        // std::cout << "init value of newton is not valid. lambda: " << lam << ", k: " << k << ", M: " << M << ", pi: "<< pi <<endl;
                        // continue;
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
                    // tvaclm_calc_metrix(sortedfcv,sortedxcv, train_x, train_y, mae, acc, prob, q, b,  M);

                    double valprob, valacc, valmae;
                    tvaclm_calc_metrix(sortedfcv,sortedxcv, val_x, val_y, valmae, valacc, valprob, q, b,  M);
                    maesum[l]+=valmae;
                    itersum[l]+=iteration;

                    std::cout << ">" << flush;
                }
            }
            std::cout << endl;
            double min_mae_ave = 1e100;
            double best_lam = -1;
            for (int l=0; l < lamn; l++){
                double lam = lam_list[l];
                double mae_ave = maesum[l]/local_cvnum;
                double iter_ave = itersum[l]/local_cvnum;
                std::cout << "          lambda: " << lam << ", iter: " << iter_ave << ", mean MAE: " << mae_ave << endl;
                if (mae_ave <= min_mae_ave) {
                    min_mae_ave = mae_ave;
                    best_lam = lam;
                }
            }
            if (best_lam < 0) std::cout << "ERROR: LAMBDA is wrong value";

            vector<vector<double>> f(d, vector<double>(y.size()));

            std::cout << "lambda: " << best_lam << ", pi: " <<pi<< ", M: " <<M<<"\n";
            double b[q - 1];
            for (int i=0; i < q-1; i++){
                b[i] =  2 * pi * i - pi * (q - 2);
            }

            chrono::system_clock::time_point  start, end; // 型は auto で可
            start = chrono::system_clock::now();
            int invalid = train_tvaclm(x, f, y, q, best_lam, b, 0.3, M, iteration);
            end = chrono::system_clock::now();
            duration = chrono::duration_cast<std::chrono::microseconds>(end-start).count()/1e6; 
            if (invalid) {
                std::cout << "++++++++++++++++++++++++++++++++++++\n    invalid init value\n+++++++++++++++++++++++++++++++++++++\n";
                myFile <<datanum<<","<< best_lam << "," << "train error!!!!"<< endl;
            }
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

            tvaclm_calc_metrix(sortedf,sortedx, x, y, trainmae, trainacc, trainprob, q, b,  M);
            tvaclm_calc_metrix(sortedf,sortedx, test_x, test_y, testmae, testacc, testprob, q, b,  M);
            
            // Send the column name to the stream
            
            // Send data to the stream
            myFile <<datanum<<","<< best_lam << ","<< iteration << ","<< duration<<","<<trainprob << ","<< trainacc << "," << trainmae<< "," <<testprob<< "," <<testacc<< "," <<testmae << endl;
            std::cout<< "____________________________________________________\n";

            std::cout << "iter: " << iteration <<" | "<< "duration(s): " << duration<<endl;
            std::cout << "TRAIN PROB: " << trainprob <<" | "<< "TRAIN ACC: " << trainacc <<" | "<< "TRAIN MAE: " << trainmae <<endl;
            std::cout << "TEST PROB: " << testprob <<" | "<< "TEST ACC: " << testacc <<" | "<< "TEST MAE: " << testmae <<endl;

            std::cout<< "++++++++++++++++++++++++++++++++++++++++++++++++++++\n" << endl;

        }
            
            // Close the file
        myFile.close();
    }

    std::cout<< "==========================================================\n\n" << endl;

}