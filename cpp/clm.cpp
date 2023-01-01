#include "flsa_dp.hpp"
#include "clm.hpp"
#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>
#include <math.h>
#include <time.h>
#include <chrono>
#include <stdexcept>
#include <iomanip>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

const double EPS = 1e-8;
const double PRE_FIT_EPS = 1e-1;
const int ITER = 10000;
const double LOG2 = 0.693147;
const int pn = 1000;
const int pm = 100;

double sigmoid(const double x) {
    return 1 / (exp(-x) + 1);
}

double clm_w_derivative(int q, double s, double *b, int y) {
    if (y == 1){
        return 1 - sigmoid(b[0] - s);
    }
    if (y == q){
        return - sigmoid(b[q - 2] - s);
    }
    return 1 - sigmoid(b[y - 2] - s) - sigmoid(b[y - 1] - s);
}

void clm_b_derivative(int n, int q, const double *f, const double *b, const int *y, const double M, VectorXd& db, MatrixXd& hessianb) {
    double sigmoid_1, sigmoid_2, b_dif_exp, inv_b_dif_exp, gamma;
    int yi;
    db = VectorXd::Zero(q-1);
    hessianb = MatrixXd::Zero(q-1, q-1);
    
    for (int i=0; i < n; i++){
        yi = y[i];
        if (yi == q) {
            sigmoid_2 = sigmoid(b[q - 2] - f[i]);
            b_dif_exp = exp(M - b[q - 2]);
            inv_b_dif_exp = 1 / b_dif_exp;
            gamma = 1 / (inv_b_dif_exp + b_dif_exp - 2);
            db(q - 2) -= 1 - sigmoid_2 + 1 / (inv_b_dif_exp - 1);
            hessianb(q-2,q-2) += sigmoid_2 * (1 - sigmoid_2) + gamma;
        }
        else if (yi == 1) {
            sigmoid_1 = sigmoid(b[0] - f[i]);
            b_dif_exp = exp(b[0] + M);
            inv_b_dif_exp = 1 / b_dif_exp;
            gamma = 1 / (inv_b_dif_exp + b_dif_exp - 2);
            db(0) -= 1 - sigmoid_1 + 1 / (b_dif_exp - 1);
            hessianb(0, 0) += sigmoid_1 * (1 - sigmoid_1) + gamma;
        }
        else {
            sigmoid_1 = sigmoid(b[yi - 1] - f[i]);
            sigmoid_2 = sigmoid(b[yi - 2] - f[i]);
            b_dif_exp = exp(b[yi - 1] - b[yi - 2]);
            inv_b_dif_exp = 1 / b_dif_exp;
            gamma = 1 / (inv_b_dif_exp + b_dif_exp - 2);
            db(yi - 1) -= 1 - sigmoid_1 + 1 / (b_dif_exp - 1);
            db(yi - 2) -= 1 - sigmoid_2 + 1 / (inv_b_dif_exp - 1);
            hessianb(yi - 2, yi - 1) -= gamma;
            hessianb(yi - 1, yi - 2) -= gamma;
            hessianb(yi - 1, yi - 1) += sigmoid_1 * (1 - sigmoid_1) + gamma;
            hessianb(yi - 2, yi - 2) += sigmoid_2 * (1 - sigmoid_2) + gamma;
        }
    }
    return;
}


double log1pexpz(double z){
    if (z < 0) return z + log1p(exp(-z));
    else return log1p(exp(z));
}


double calc_clm_objective(vector<double> fsum, vector<int> y, double *b, int q){
    double loss;
    for (size_t i=0; i < y.size(); i++){
        if (y[i] == 1){
            loss += log1pexpz(fsum[i] - b[0]);
        }
        else if (y[i] == q){
            loss += log1pexpz(b[q - 2] - fsum[i]);
        }else {
            double delta = b[y[i] - 1] - b[y[i] - 2];
            double gamma = b[y[i] - 2] - fsum[i];
            double c1;
            if (delta < LOG2) c1 = log(-expm1(-delta));
            else c1 = log1p(-exp(-delta));
            loss -= c1 - log1pexpz(-gamma-delta) - log1pexpz(gamma);
        }
    }
    return loss;
}


vector<double> calc_clm_suboptibality(const vector<vector<double>> x, const vector<int> y, int q, double *b, vector<double> fsum){
    double subopt;
    double temp_loss;
    bool is_v_equivalent;
    int n = y.size();
    int d = x.size();
    double clmwd;
    vector<double> gradient(d+1);
    
    for (int i = 0; i < n; i++){
        clmwd = clm_w_derivative(q, fsum[i], b, y[i]);
        for (int j = 0; j < d; j++) gradient[j] += x[j][i] * clmwd;
        gradient[d] += clmwd;
    }
    return gradient;
}

int trainClm(const vector< vector<double>> x, const vector<int> y, const int q, vector<double> &w ,double *b, const double L, const double M){
    int d = x.size();
    int n = y.size();
    vector<double> fsum(n);
    double l_inv = 1/(L * n);
    double loss;
    MatrixXd hessianb(q-1,q-1);
    LLT<MatrixXd> llt;
    VectorXd newton(q-1);
    VectorXd db(q-1);
    vector<int> sorted_y(n);
    vector<double> sorted_fsum(n);
    vector<double> sorted_f(n);
    vector<double> sorted_solver_y(n);
    double duration = 0;
    double subopt;
    double init_subopt;
    vector<double> gradient;
    for (int k=0; k<10; k++) {
        for (int i = 0; i < n; i++){
            fsum[i] = 0;
            for (int j = 0; j < d; j++) fsum[i] += w[j] * x[j][i];
            fsum[i] += w[d];
        }
        gradient = calc_clm_suboptibality(x,y,q,b, fsum);
        for (int j = 0; j <= d; j++) w[j] -= l_inv * gradient[j];

    }
    for (int k=0; k<ITER; k++) {
        subopt= 0;
        chrono::system_clock::time_point  start, end; // 型は auto で可
        start = chrono::system_clock::now();
        clm_b_derivative(n, q, &fsum[0], b, &y[0], M, db, hessianb);
        llt.compute(hessianb);
        newton = llt.solve(db);
        for (int l = 0; l<q-1; l++) b[l] -= newton[l];
        if (abs(b[0]) > M) {
            std::cout << "invalid newton initial value" <<endl;
            return 1;
        }
        for (int i = 0; i < n; i++){
            fsum[i] = 0;
            for (int j = 0; j < d; j++) fsum[i] += w[j] * x[j][i];
            fsum[i] += w[d];
        }
        gradient = calc_clm_suboptibality(x,y,q,b, fsum);
        subopt = 0;
        for (int i = 0; i < d + 1; i++) subopt += abs(gradient[i]);
        for (int i = 0; i < q-1; i++) subopt += abs(db[i]);
        if (k==0) init_subopt = subopt;
        end = chrono::system_clock::now();
        duration += chrono::duration_cast<std::chrono::microseconds>(end-start).count(); 
        if (k%pn==0){
            if (b[0] < -M || b[q-2] > M) return 1;
            for (int l = 0; l<q-2; l++) if (b[l+1] - b[l] < 0) return 1;
            loss = calc_clm_objective(fsum, y, b, q);
            if (loss==INFINITY) return loss;
            cout << "iter: " << k;
            cout << " loss: " << loss;
            cout << " subopt: "<<subopt << "\n";
            cout << "b: ";
            for (int j=0; j<q-1; j++) cout << b[j] << " ";
            cout << endl;
            cout << "w: ";
            for (int j=0; j<=d; j++) cout << w[j] << " ";
            cout <<endl;
            
        }
        if (subopt/init_subopt < EPS) {
            std::cout << "converged: " << k << "\n";

            cout << "w: ";
            for (int j=0; j<=d; j++) cout << w[j] << " ";
            cout <<endl;
            break;
        }
        for (int j = 0; j <= d; j++) w[j] -= l_inv * gradient[j];
    }
    return 0;
}