#include "flsa_dp.hpp"
#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>
#include <math.h>

using namespace std;

double clm_derivative(int q, double x, double *b, int y);
void solve_square(const int n, const int q, double *f, double *b, double *y, double *solver_y);
void set_argsort(vector<vector<int>> &argsort, vector<deque<bool>> &argsort_c, vector< vector<int>> &argsort_inv, vector<vector<double>> const &x);
void solve_block_coordinate_descent(vector<vector<double>> x, vector<vector<double>>& f, vector<int> y , int q, double lam, double *b);
void solve_gradient_descent(vector< vector<double>> x, vector< vector<double>>& f, vector<int> y , int q, double lam, double *b);