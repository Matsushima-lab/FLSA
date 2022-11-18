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
    string dataname = "car";
    std::string filename = "/home/iyori/work/gam/experiments/datasets/ordinal-regression/"+dataname+"/matlab/train_"+dataname+".0";
    // std::string filename = "/home/iyori/work/gam/experiments/datasets/discretized-regression/10bins/"+dataname+"/matlab/train_"+dataname+".1";
    // std::string filename = "/home/iyori/work/gam/experiments/datasets/ordinal-regression/winequality-red/matlab/train_winequality-red.1";
    // std::string filename = "/home/iyori/work/gam/experiments/datasets/ordinal-regression/automobile/matlab/train_automobile.0";
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
        y.push_back((int) data[i][d]);
    }

    // vector<vector<double>> x{{0,1,3,3,4}};
// , {8,3,0,2,3,4,1,1,2,1}

    // vector<vector<double>> x{{4,1,3,6,2,7,4,9,0,2},{6,8,3,8,0,4,7,1,8,9},{1,2,5,3,3,0,5,9,2,6}};

    // vector<vector<double>> x{{0,1,1}};
    // vector<int> y{1,2,5,3,1,1,3,2,2,4};

    // vector<int> y{2,2,1};

    int q = *max_element(y.begin(), y.end());
    double b[q];
    std::iota(b, b+q, 1);
    std::cout << n << ", " << d << "\n";
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

    solve_block_coordinate_descent(x, f, y, q, lam, b, 0.2);
    
    // for (int j = 0; j < d; j++){
    //     print(f[j]);
    // }
    // double min_loss = INFINITY;
    // double loss;
    // double minL, mint0;
    // cout << "_____________________________________________\n";
    // // for (int i=-5; i<=5; i++){
    // //     for (int j=-4; j<=5; j++){
    //         double L = 1000; //* pow(2,i);
    //         double t0 = 100; //* pow(4,j);
    //         fill(f1.begin(), f1.end(), vector<double>(n));
    //         loss =  solve_gradient_descent(x, f1, y, q, lam, b, L, t0);
    //         cout << "L: "<< L << ", t0: "<< t0<< "  final loss:" << loss << "\n";
    //         if (min_loss>loss){
    //             min_loss = loss;
    //             minL = L;
    //             mint0 = t0;
    //         }
    // //     }
    // // }
    // cout << "min_L: " << minL<< " , min_t0: " << mint0<<  "  min loss: " <<min_loss <<" \n";
    // for (int j = 0; j < d; j++){
    //     print(f1[j]);
    // }


}