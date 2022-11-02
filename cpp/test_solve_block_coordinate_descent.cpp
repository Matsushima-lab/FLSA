#include "flsa_dp.hpp"
#include "solve_block_coordinate_descent.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;

template <typename T>
void print(std::vector<T> const &input)
{
    for (int i = 0; i < input.size(); i++) {
        std::cout << input.at(i) << ' ';
    }
    std::cout << '\n' << ' ';
}

// void print(std::vector<bool> const &input)
// {
//     for (int i = 0; i < input.size(); i++) {
//         std::cout << input.at(i) << ' ';
//     }
// }

int main(){
    vector<vector<double>> x{{0,3,2,4,1},{2,3,4,1,2},{2,3,3,3,2}};
    size_t n = x[0].size();
    size_t d = x.size();
    vector<double> fsum(n, 0);
    vector<double> fsumtmp(n, 0);
    vector<vector<size_t>> argsort(d, vector<size_t>(n));
    vector<vector<bool>> argsort_c(d, vector<bool>(n-1, 0));
    vector<vector<size_t>> argsort_inv(d, vector<size_t>(n));

    set_argsort(argsort, argsort_c, argsort_inv, x);

    for (size_t j = 0; j < d; j++){
        print(argsort[j]);
    }

    for (size_t j = 0; j < d; j++){
        print(argsort_c[j]);
    }

    for (size_t j = 0; j < d; j++){
        print(argsort_inv[j]);
    }



    

}