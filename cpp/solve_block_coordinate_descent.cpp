#include "flsa_dp.hpp"
#include "solve_block_coordinate_descent.hpp"
#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>
#include <math.h>
#include <time.h>
#include <chrono>

using namespace std;
// L must be less than 1
const double L = 0.25;
const double EPS = 1e-6;
const int ITER = 20;
const int pn = 1;
double l_inv = 1/(L);
double t0 = 100;

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


void solve_square(const int n, const int q, double *fsum, double* f, double *b, int *y, double *solver_y, int iter) {
    double grad;
    cout << "solever y :  ";

    // calc solver_y
    for (int i = 0; i < n; i++){
        grad = clm_derivative(q, fsum[i], b, y[i]);
        solver_y[i] = f[i] - l_inv * grad;
        cout << clm_derivative(q, fsum[i], b, y[i])<< ";"<<solver_y[i] << ":" << f[i]<<" ";
    }
    cout << "\n";
    return;
}

double calc_loss(vector<double> fsum, vector<int> y, double *b, int q, vector<vector<double>> f, vector<vector<int>> argsort, double lam){
    double loss;
    // cout << "loss";
    for (size_t i=0; i < y.size(); i++){
        if (y[i] == 1){
            loss -= log(sigmoid(b[1] - fsum[i]));
        }
        else if (y[i] == q){
            loss -= log(1 - sigmoid(b[q - 1] - fsum[i]));
        }else {
            loss -= log(sigmoid(b[y[i] - 1] - fsum[i]) - sigmoid(b[y[i] - 2] - fsum[i]));
        }
        // cout << loss << " " << fsum[i];
    }
    // cout << "\n";
    for (size_t j=0; j < f.size() - 1; j++){
        for (size_t i=0; i < y.size() - 1; i++){
            loss += lam * abs(f[j][argsort[j][i+1]] - f[j][argsort[j][i]]);
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
double calc_min_subopt(double loss, double lam){
    if (loss > lam) {
        return loss - lam;
    } else if (loss < -lam) {
        return loss + lam;
    } else{
        return 0;
    }
}

double calc_suboptibality(int q, int n, double *b, double *fsum, double *f, double lam, bool *c, int *y, double *gradient){
    double subopt;
    double temp_loss;
    bool is_v_equivalent;
    int *gsign;
    gsign = new int[n + 1];
    gsign[0] = 0;
    gsign[n] = 0;
    // cout << "gsign: ";
    for (int i = 1; i < n;i++) {
        if (f[i - 1] > f[i]){
            gsign[i] = 1;
        }
        // else if (abs(f[i - 1]-f[i]) < EPS/n){
        else if (f[i - 1]==f[i]){
            gsign[i] = 2;
        }
        else {
            gsign[i] = -1;
        }
        // cout << gsign[i] << ",";
    }
    // cout << "\n";
    int pre_i = 0;
    cout << "temp_loss: " << endl;
    for (int i = 0; i < n; i++){
        temp_loss += clm_derivative(q, fsum[i], b, y[i]);
        cout << temp_loss << " -> ";
        if (!c[i]) {
            if (gsign[pre_i] < 2) {
                temp_loss -= gsign[pre_i] * lam;
            }
            if (gsign[i + 1] < 2) {
                temp_loss += gsign[i + 1] * lam;
            }
            if (gsign[pre_i] == 2 && gsign[i + 1] == 2) {
                temp_loss = calc_min_subopt(temp_loss, 2 * lam);
            } else if (gsign[pre_i] == 2 || gsign[i + 1] == 2) {
                temp_loss = calc_min_subopt(temp_loss, lam);
            }
            cout << temp_loss << " ";
            // cout << pre_i <<"-"<< i << ";   ";
            subopt += abs(temp_loss);
            for (int j = pre_i; j <= i; j++){
                gradient[j] = temp_loss;

                // cout <<j <<";"<<gradient[j] << " ";
            }
            pre_i = i + 1;
            temp_loss = 0;
        }
    }
    cout << "\n";
    return subopt;
}




void solve_block_coordinate_descent(vector< vector<double>> x, vector< vector<double>>& f, vector<int> y , int q, double lam, double *b) {
    double loss;
    double lastloss;
    int n = y.size();
    int d = x.size();
    double duration = 0;
    vector<double> fsum(n, 0);
    vector<double> fsumtmp(n, 0);
    vector<vector<int>> argsort(d, vector<int>(n));
    vector<deque<bool>> argsort_c(d,deque<bool>(n));
    vector<vector<int>> argsort_inv(d, vector<int>(n));
    set_argsort(argsort, argsort_c, argsort_inv, x);
    vector<double> solver_y(n);
    vector<double> solver_f(n);
    vector<double> sorted_solver_y(n);
    vector<int> sorted_y(n);
    vector<double> sorted_fsum(n);
    vector<double> sorted_f(n);
    vector<vector<double>> gradient(d, vector<double>(n));
    double flsa_lam = l_inv * lam * 0.5;

    double subopt;

    // while (1) {

    for (int k=0; k<ITER; k++) {
        chrono::system_clock::time_point  start, end; // 型は auto で可
        start = chrono::system_clock::now();
        for (int j = 0; j < d; j++){
            // cout << "fsum: ";
            // for (int i = 0; i < n; i++){
            //     cout<<fsum[i]<<" ";
            // }
            // cout << "\n";

            solve_square(n, q, &fsum[0], &f[j][0], b, &y[0], &solver_y[0], k);

            for (int i = 0; i < n; i++){
                fsumtmp[i] = fsum[i] - f[j][i];
                sorted_solver_y[i] = solver_y[argsort[j][i]];
            }

            tf_dp(n, &sorted_solver_y[0], flsa_lam, &argsort_c[j][0], &solver_f[0]);

            // cout << "f: ";
            for (int i = 0; i < n; i++){
                f[j][i] = solver_f[argsort_inv[j][i]];
                fsum[i] = fsumtmp[i] + f[j][i];
                // cout<<f[j][i]<<" ";
            }
            // cout << "\n";
        }
        end = chrono::system_clock::now();
        duration += chrono::duration_cast<std::chrono::microseconds>(end-start).count(); 
        subopt = 0;
        for (int j = 0; j < d; j++){
            for (int i = 0; i < n; i++){
                // これは毎回やる必要はない
                sorted_y[i] = y[argsort[j][i]];
                sorted_fsum[i] = fsum[argsort[j][i]];
                sorted_f[i] = f[j][argsort[j][i]];
            }
            subopt += calc_suboptibality(q, n, b, &sorted_fsum[0], &sorted_f[0],lam, &argsort_c[j][0], &sorted_y[0], &gradient[j][0]);
        }
        double loss =  calc_loss(fsum, y, b, q, f, argsort, lam);
        // cout << "subopt" << " : "<<subopt << "\n";
        if (k%pn==0){
            cout <<"loss: " << loss <<" , subopt" << " : "<<subopt << "\n";
        }
        if (subopt < EPS) {
            cout << "converged: " << k << "\n";
            cout << duration << "\n";
            break;
        }
        lastloss = loss;

    }

    cout << "duration: " << duration << "\n";
}




void solve_gradient_descent(vector< vector<double>> x, vector< vector<double>>& f, vector<int> y , int q, double lam, double *b) {
    double loss;
    double lastloss;
    int n = y.size();
    int d = x.size();
    double subopt;
    vector<vector<double>> updated_f(d, vector<double>(n));
    vector<double> fsum(n, 0);
    vector<double> fsumtmp(n, 0);
    vector<vector<int>> argsort(d, vector<int>(n));
    vector<deque<bool>> argsort_c(d,deque<bool>(n));

    vector<vector<int>> argsort_inv(d, vector<int>(n));
    set_argsort(argsort, argsort_c, argsort_inv, x);
    vector<double> solver_y(n);
    vector<double> solver_f(n);
    double tmp_f;

    vector<int> sorted_y(n);
    vector<double> sorted_fsum(n);
    vector<double> sorted_f(n);
    vector<double> sorted_solver_y(n);
    double duration = 0;

   
    vector<vector<double>> gradient(d, vector<double>(n));
    for (int k=0; k<ITER; k++) {
        subopt= 0;
        // cout << "fsum: ";

        chrono::system_clock::time_point  start, end; // 型は auto で可
        start = chrono::system_clock::now();
        for (int i = 0; i < n; i++){

            fsum[i] = 0;
            for (int j = 0; j < d; j++){
                fsum[i] += f[j][i];
            }
            // cout << fsum[i] << ", ";
        }
        // cout << "\n";
        for (int j = 0; j < d; j++){
            for (int i = 0; i < n; i++){
                sorted_y[i] = y[argsort[j][i]];
                sorted_fsum[i] = fsum[argsort[j][i]];
                sorted_f[i] = f[j][argsort[j][i]];
            }
            subopt += calc_suboptibality(q, n, b, &sorted_fsum[0], &sorted_f[0],lam, &argsort_c[j][0], &sorted_y[0], &gradient[j][0]);
        }

        end = chrono::system_clock::now();
        duration += chrono::duration_cast<std::chrono::microseconds>(end-start).count(); 

        double loss =  calc_loss(fsum, y, b, q, f, argsort, lam);
        if (k%pn==0){
            cout << "loss: " << loss;
            cout << " . subopt" << " : "<<subopt << "\n";
        }
        
        if (subopt < EPS) {
            cout << "converged: " << k << "\n";
            break;
        }

        for (int j = 0; j < d; j++){
            for (int i = 0; i < n; i++){
                // updated_f[j][i]  = f[j][i] - l_inv * gradient[j][argsort_inv[j][i]];
                // if (i!=0 && (updated_f[j][i] - updated_f[j][i-1]) * (f[j][i] -f[j][i-1]) <=0 ) {
                //     updated_f[j][i] = updated_f[j][i-1];
                // }
                // f[j][i] -= updated_f[j][i];
                
                // if (i!=n-1 && (f[j][i + 1]-f[j][i])* (f[j][i + 1] -f[j][i] + tmp_f) <=0 ) {
                //     f[j][i] = f[j][i+1];
                //     continue;
                // }

                f[j][i] -= l_inv / sqrt(t0 + k) * gradient[j][argsort_inv[j][i]];
                // cout <<i<<argsort_inv[j][i]<<gradient[j][argsort_inv[j][i]]<<f[j][i] << " ";
                // cout <<f[j][i] << " ";
            }
            // cout <<  "\n";
        }
        // cout << "loss: " << loss;
        // cout << " . subopt" << " : "<<subopt << "\n";
        
    }

    cout << "duration: " << duration << "\n";
}