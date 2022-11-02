#include "flsa_dp.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;

double clm_derivative(size_t q, double x, double *b, size_t y);
void solve_square(size_t n, size_t q, double *f, double *b, double *y, double *fj);
void set_argsort(vector<vector<size_t>> &argsort, vector<vector<bool>> &argsort_c, vector< vector<size_t>> &argsort_inv, vector<vector<double>> const &x);
void solve_block_coordinate_descent(vector<vector<double>> x, vector<vector<double>> f, vector<double> y , size_t q, double lam);