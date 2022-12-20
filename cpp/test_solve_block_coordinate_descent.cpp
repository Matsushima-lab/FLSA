#include "flsa_dp.hpp"
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

template <typename T>
void print(std::vector<T> const &input)
{
    for (int i = 0; i < input.size(); i++) {
        std::cout << input.at(i) << ' ';
    }
    std::cout << '\n';
}

//文字列のsplit機能
std::vector<std::string> split(std::string str, char del) {
    int first = 0;
    int last = str.find_first_of(del);
    std::vector<std::string> result;
    while (first < str.size()) {
        std::string subStr(str, first, last - first);
        result.push_back(subStr);
        first = last + 1;
        last = str.find_first_of(del, first);
        if (last == std::string::npos) {
            last = str.size();
        }
    }
    return result;
}

std::vector<std::vector<double> >
csv2vector(std::string filename, int ignore_line_num = 0){
    //csvファイルの読み込み
    std::ifstream reading_file;
    reading_file.open(filename, std::ios::in);
    if(!reading_file){
        std::vector<std::vector<double> > data;
        return data;
    }
    std::string reading_line_buffer;
    //最初のignore_line_num行を空読みする
    for(int line = 0; line < ignore_line_num; line++){
        getline(reading_file, reading_line_buffer);
        if(reading_file.eof()) break;
    }

    //二次元のvectorを作成
    std::vector<std::vector<double> > data;
    std::vector<double> row{};
    while(std::getline(reading_file, reading_line_buffer)){
        data.push_back(vector<double> {});
        if(reading_line_buffer.size() == 0) break;
        std::vector<std::string> temp_data;
        temp_data = split(reading_line_buffer, ' ');
        for (auto const& i : temp_data){
            data.back().push_back(std::stod(i));
        }
    }
    return data;
}

template <typename T>
void print(std::vector<T> const &input, std::vector<int> const &index)
{
    for (int i = 0; i < input.size(); i++) {
        std::cout << input.at(index.at(i)) << ' ';
    }
    std::cout << '\n';
}

// void print(std::vector<bool> const &input)
// {
//     for (int i = 0; i < input.size(); i++) {
//         std::cout << input.at(i) << ' ';
//     }
// }
// -2.08495 0.499999 1.37134 0.286575 -1.37134 

int main(){
    double yi;

    string dataname = "wlb";
    std::string filename = "/home/iyori/work/gam/datasets/"+dataname+"/train_"+dataname+".2";
    std::string test_filename = "/home/iyori/work/gam/datasets/"+dataname+"/test_"+dataname+".2";


    // string dataname = "winequality-red";
    // std::string filename = "/home/iyori/work/gam/experiments/datasets/ordinal-regression/"+dataname+"/matlab/train_"+dataname+".5";
    // std::string test_filename = "/home/iyori/work/gam/experiments/datasets/ordinal-regression/"+dataname+"/matlab/test_"+dataname+".5";

    // std::string filename = "/home/iyori/work/gam/experiments/datasets/discretized-regression/5bins/"+dataname+"/matlab/train_"+dataname+".1";

    vector<vector<double>> train_data = csv2vector(filename);
    int d = train_data[0].size() - 1;
    // int n = train_data.size();
    int n = train_data.size();
    double M = 10;
    vector<vector<double>> x(d,vector<double>{});
    vector<int> y{};
    for (int i=0; i<n; i++){
        // if (i.size()!=d+1){
        //     throw std::invalid_argument("received negative value");
        // }
        for (size_t j=0; j<d; j++){
            x[j].push_back(train_data[i][j]);
        }
        y.push_back((int) train_data[i][d]);
    }
    // read test data;
    vector<vector<double>> test_data = csv2vector(test_filename);
    int test_n = test_data.size();
    vector<vector<double>> test_x(d,vector<double>{});
    vector<int> test_y{};
    for (int i=0; i<test_n; i++){
        // if (i.size()!=d+1){
        //     throw std::invalid_argument("received negative value");
        // }
        for (size_t j=0; j<d; j++){
            test_x[j].push_back(test_data[i][j]);
        }
        test_y.push_back((int) test_data[i][d]);
    }
    std::cout << "test_n: " << test_n << ", n: " << n << "\n";

    int q = *max_element(y.begin(), y.end());
    double b[q - 1];
    std::cout << "b: ";
    for (int i=0; i < q-1; i++){
        // b[i] = 2 * M / q * (i + 1) - M;
        b[i] = 4 * i - 2 * (q - 2);
        std::cout << b[i] << " ";
    }
    std::cout << "\n";
    double lam = 0.4;
    vector<vector<double>> f(d, vector<double>(n));
    vector<vector<double>> f1(d, vector<double>(n));
    vector<vector<double>> sortedf(d, vector<double>(n));
    vector<vector<double>> sortedx(d, vector<double>(n));
    vector<double> fsum(n, 0);
    vector<double> fsumtmp(n, 0);
    vector<vector<int>> argsort(d, vector<int>(n));
    vector<vector<int>> argsort_c(d, vector<int>(n-1));
    vector<vector<int>> argsort_inv(d, vector<int>(n));

    set_argsort(argsort, argsort_c, argsort_inv, x);

    // for (int j = 0; j < d; j++){
    //     print(y, argsort[j]);
    // }
    // for (int j = 0; j < d; j++){
    //     for (int i = 0; i < n - 1; i++){
    //         cout << argsort_c[j][i] << " ";
    //     }
    //     cout << "\n";
    // }
    // for (int j = 0; j < d; j++){
    //     print(argsort_inv[j]);
    // }

    solve_block_coordinate_descent(x, f, y, q, lam, b, 0.3, M);
    std::cout <<"_______________________________________________\n";
    std::cout << "b: ";
    for (int l = 0; l<q-1; l++){
        std::cout << b[l] << " ";
    }
    std::cout << "\n";

    std::cout <<"_______________________________________________\n";
    double f_train;
    double qprob, qprobmax, tmp;
    int pred;
    double prob = 0;
    double acc = 0;
    double mae = 0;

    
    for (int k = 0; k < n; k++){
        tmp = 0;
        for (int j = 0; j< d;j++){
            tmp += f[j][k];
        }
    }

    for (int j = 0; j< d;j++){
        for (int k = 0; k < n; k++){
            sortedf[j][k] = f[j][argsort[j][k]];
            sortedx[j][k] = x[j][argsort[j][k]];
        }
    }

    for (int k = 0; k < n; k++){
        qprobmax = 0;
        f_train = 0;
        pred = 0;

        for (int j = 0; j< d; j++){
            // cout << x[j][argsort[j][n-1]] << ':' << test_x[j][k]<< " ";
            // cout << x[j][k] << ":" << sortedx[j][n-1] << "; "<< f_train << " # " ;
            if (sortedx[j][n-1] < x[j][k]) f_train+=sortedf[j][n-1];
            else{

                for (int i = 0; i< n;i++){
                    if (sortedx[j][i] >= x[j][k]) 
                    {
                        f_train+=sortedf[j][i];
                        // cout << sortedf[j][i] << " || ";
                        break;
                    }
                }
            }
        }
        // cout << "\n" << f_train << "\n";
        for (int l = 1; l<=q; l++){
            if (l==1) qprob = sigmoid(b[0] - f_train) - sigmoid(- f_train);
            else if (l<q) qprob = sigmoid(b[l - 1] - f_train) - sigmoid(b[l - 2]- f_train);
            else qprob = sigmoid(M - f_train) - sigmoid(b[q - 2]- f_train);
            if (l == y[k]) prob += qprob;
            if (qprobmax < qprob){
                qprobmax = qprob;
                pred = l;
            }
        }

        // if (k < 5) cout << f_train << ": " << pred << "; " <<y[k]<<"\n______________________________________________\n";
        // cout << "f: " << f_train << ", pred: " << pred << ", y: " << y[k] << "\n";

        if (pred==y[k]) acc+=1;
        mae+=abs(pred-y[k]);
    }
    std::cout << "PROB: " << prob/n <<"\n";
    std::cout << "ACC: " << acc/n <<"\n";
    std::cout << "MAE: " << mae/n <<"\n";


    double f_test;
    // double qprob, qprobmax;
    // int pred;
    acc = 0;
    mae = 0;
    prob = 0;
    for (int k = 0; k < test_n; k++){
        qprobmax = 0;
        f_test = 0;
        pred = 0;

        for (int j = 0; j< d; j++){
            if (sortedx[j][n-1] < test_x[j][k]) f_test+=sortedf[j][n-1];
            else{

                for (int i = 0; i< n;i++){
                    if (sortedx[j][i] >= test_x[j][k]) 
                    {
                        f_test+=sortedf[j][i];
                        break;
                    }
                }
            }
        }
        for (int l = 1; l<=q; l++){
            if (l==1) qprob = sigmoid(b[0] - f_test) - sigmoid(- f_test);
            else if (l<q) qprob = sigmoid(b[l - 1] - f_test) - sigmoid(b[l - 2]- f_test);
            else qprob = sigmoid(M - f_test) - sigmoid(b[q - 2]- f_test);
            if (l == test_y[k]) prob += qprob;
            if (qprobmax < qprob){
                qprobmax = qprob;
                pred = l;
            }
        }
        if (pred==test_y[k]) acc+=1;
        mae += std::abs(pred - test_y[k]);
    }
    std::cout << "PROB: " << prob/test_n <<"\n";
    std::cout << "ACC: " << acc/test_n <<"\n";
    std::cout << "MAE: " << mae/test_n <<"\n";


}