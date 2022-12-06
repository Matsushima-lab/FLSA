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
    // std::string filename = "/home/iyori/work/gam/experiments/datasets/ordinal-regression/"+dataname+"/matlab/train_"+dataname+".1";
    // std::string filename = "/home/iyori/work/gam/experiments/datasets/discretized-regression/5bins/"+dataname+"/matlab/train_"+dataname+".1";

    vector<vector<double>> data = csv2vector(filename);
    int d = data[0].size() - 1;
    int n = data.size();
    vector<vector<double>> x(d,vector<double>{});
    vector<int> y{};
    for (int i=0; i<n; i++){
        // if (i.size()!=d+1){
        //     throw std::invalid_argument("received negative value");
        // }
        for (size_t j=0; j<d; j++){
            x[j].push_back(data[i][j]);
        }
        y.push_back((int) data[i][d] + 1);
    }

    int q = *max_element(y.begin(), y.end());
    double b[q - 1];
    std::iota(b, b + q - 1, -(double)q/2.0+1);
    double lam = 0.1;
    vector<vector<double>> f(d, vector<double>(n));
    vector<vector<double>> f1(d, vector<double>(n));
    vector<double> fsum(n, 0);
    vector<double> fsumtmp(n, 0);
    vector<vector<int>> argsort(d, vector<int>(n));
    vector<vector<int>> argsort_c(d, vector<int>(n-1));
    vector<vector<int>> argsort_inv(d, vector<int>(n));

    // set_argsort(argsort, argsort_c, argsort_inv, x);

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

    solve_block_coordinate_descent(x, f, y, q, lam, b, 0.5);
    std::cout <<"_______________________________________________\n";
    std::cout << "b: \n";
    for (int l = 0; l<q-1; l++){
        std::cout << b[l] << " ";
    }
    std::cout << "\n";
 
    std::cout << "f: \n";
    for (int j = 0; j<d; j++){
        for (int i = 0; i<n; i++){
        std::cout << f[j][i] << " ";
        }
        std::cout << "\n";
    }

    
    double min_loss = INFINITY;
    double loss;
    double minL, mint0;
 

}