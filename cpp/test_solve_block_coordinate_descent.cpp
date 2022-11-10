#include "flsa_dp.hpp"
#include "solve_block_coordinate_descent.hpp"
#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>
#include <math.h>

using namespace std;

template <typename T>
void print(std::vector<T> const &input)
{
    for (int i = 0; i < input.size(); i++) {
        std::cout << input.at(i) << ' ';
    }
    std::cout << '\n';
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
    // vector<vector<double>> x{{0,1,3,3,4}};
    // vector<vector<double>> x{{1,3,2,3,5},{1,2,3,1,2}};
    vector<vector<double>> x{{1,2,3,4,3}};
    // -1.37134 0.404811 -1.37134 
    vector<int> y{1,3,2,2,1};
    double b[] = {0, 1};
    int q = 3;
    int n = x[0].size();
    int d = x.size();
    double lam = 0.05;
    vector<vector<double>> f(d, vector<double>(n));
    vector<vector<double>> f1(d, vector<double>(n));
    vector<double> fsum(n, 0);
    vector<double> fsumtmp(n, 0);
    vector<vector<int>> argsort(d, vector<int>(n));
    vector<deque<bool>> argsort_c(d, deque<bool>(n-1));
    vector<vector<int>> argsort_inv(d, vector<int>(n));

    set_argsort(argsort, argsort_c, argsort_inv, x);

    for (int j = 0; j < d; j++){
        print(y, argsort[j]);
    }

    for (int j = 0; j < d; j++){
        for (int i = 0; i < n - 1; i++){
            cout << argsort_c[j][i] << " ";
        }
        cout << "\n";
    }

    for (int j = 0; j < d; j++){
        print(argsort_inv[j]);
    }

    solve_block_coordinate_descent(x, f, y, q, lam, b);
    
    for (int j = 0; j < d; j++){
        print(f[j]);
    }

    // cout << "_____________________________________________\n";
    // solve_gradient_descent(x, f1, y, q, lam, b);
    // for (int j = 0; j < d; j++){
    //     print(f1[j]);
    // }


}