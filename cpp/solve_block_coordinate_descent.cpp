#include "flsa_dp.hpp"
#include "solve_block_coordinate_descent.hpp"
#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>
#include <math.h>

using namespace std;

const double L = 0.5;
const double EPS = 1e-7;

double sigmoid(double x) {
    return 1 / (exp(-x) + 1);
}

double clm_derivative(int q, double x, double *b, int y) {
    if (y == 1){
        return 1 - sigmoid(b[0] - x);
    }
    if (y == q){
        return - sigmoid(b[q - 1] - x);
    }
    return 1 - sigmoid(b[y - 2] - x) - sigmoid(b[y - 1] - x);
}

void solve_square(int n, int q, double *f, double *b, int *y, double *solver_y) {
    double l_inv = 1/L;

    // calc solver_y
    for (int i = 0; i < n; i++){
        solver_y[i] = l_inv * clm_derivative(q, f[i], b, y[i]);
    }
    return;
}

double calc_loss(vector<double> fsum, vector<int> y, double *b, int q){
    double loss;
    for (size_t i=0; i < y.size(); i++){
        if (y[i] == 1){
            loss -= log(sigmoid(b[1] - fsum[i]));
        }
        else if (y[i] == q){
            loss -= log(1 - sigmoid(b[q - 1] - fsum[i]));
        }else {
            loss -= log(sigmoid(b[y[i] - 1] - fsum[i]) - sigmoid(b[y[i] - 2] - fsum[i]));
        }
    }
    return loss;
}
    

void set_argsort(vector<vector<int>> &argsort, vector<deque<bool>> &argsort_c, vector<vector<int>> &argsort_inv, vector<vector<double>> const &x){
    int d = x.size();
    int n = x[0].size();
    int i;
    for (int j = 0; j<d; j++){
        vector<double> xj = x[j];
        argsort[j];
        iota(argsort[j].begin(), argsort[j].end(), 0);
        sort(argsort[j].begin(), argsort[j].end(),
            [&xj](int left, int right) -> bool {
                  // sort indices according to corresponding array element
                  return xj[left] < xj[right];
              });
        for (i = 0; i < n; i++) {
            if (i!=n-1 && xj[argsort[j][i]]==xj[argsort[j][i + 1]]){
                argsort_c[j][i] = 1;
            }
            argsort_inv[j][argsort[j][i]] = i;
        }
    }
    return;
}


void solve_block_coordinate_descent(vector< vector<double>> x, vector< vector<double>>f, vector<int> y , int q, double lam) {
    double b[] = {-1, 0, 1};
    double loss;
    double lastloss;
    int n = y.size();
    int d = x.size();
    vector<double> fsum(n, 0);
    vector<double> fsumtmp(n, 0);
    vector<vector<int>> argsort(d, vector<int>(n));
    vector<deque<bool>> argsort_c(d,deque<bool>(n));
    vector<vector<int>> argsort_inv(d, vector<int>(n));
    set_argsort(argsort, argsort_c, argsort_inv, x);
    vector<double> solver_y(n);
    vector<double> solver_f(n);
    vector<double> sorted_solver_y(n);

    while (1) {
        for (int j = 0; j < d; j++){
            
            solve_square(n, q, &fsum[0], b, &y[0], &solver_y[0]);
            for (int i = 0; i < n; i++){
                fsumtmp[i] = fsum[i] - f[j][i];
                sorted_solver_y[i] = solver_y[argsort[j][i]];
            }
            tf_dp(n, &sorted_solver_y[0], lam, &argsort_c[j][0], &solver_f[0]);
            for (int i = 0; i < n; i++){
                f[j][i] = solver_f[argsort_inv[j][i]];
                fsum[i] = fsumtmp[i] + f[j][i];
            }
            loss = calc_loss(fsum, y, b, q);
            if ((loss - lastloss) * (loss - lastloss) < EPS) {
                break;
            }
            lastloss = loss;
        }
    }
}
