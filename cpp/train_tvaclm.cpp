#include "flsa_dp.hpp"
#include "train_tvaclm.hpp"
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

const double EPS = 1e-10;
const double PRE_FIT_EPS = 1e-1;
const int ITER = 1000000;
const double LOG2 = 0.693147;
const int pn = 1000;
const int pm = 10;



void solve_square(const int n, const int q, double *fsum, double* f, double *b, const int *y, double *solver_y, int iter, const double l_inv) {
    double grad;

    for (int i = 0; i < n; i++){
        grad = clm_w_derivative(q, fsum[i], b, y[i]);
        solver_y[i] = f[i] - l_inv * grad;
    }
    return;
}

double calc_objective(vector<double> fsum, vector<int> y, double *b, int q, vector<vector<double>> f, vector<vector<int>> argsort, double lam){

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
    for (size_t j=0; j < f.size(); j++){
        for (size_t i=0; i < y.size() - 1; i++){
            loss += lam * abs(f[j][argsort[j][i+1]] - f[j][argsort[j][i]]);
        }
    }
    return loss;
}
    

void set_argsort(vector<vector<int>> &argsort, vector<vector<int>> &argsort_c, vector<vector<int>> &argsort_inv, vector<vector<double>> const &x){
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

double calc_suboptibality(int q, int n, double *b, double *fsum, double *f, double lam, const vector<int> c, int *y, double *gradient){
    double subopt;
    double temp_loss;
    bool is_v_equivalent;
    int *gsign;
    gsign = new int[n + 1];
    gsign[0] = 0;
    gsign[n] = 0;
    for (int i = 1; i < n;i++) {
        if (f[i - 1] > f[i]){
            gsign[i] = 1;
        }
        else if (f[i - 1]==f[i]){
            gsign[i] = 2;
        }
        else {
            gsign[i] = -1;
        }
    }
    int pre_i = 0;
    for (int i = 0; i < n-1; i++){
        temp_loss += clm_w_derivative(q, fsum[i], b, y[i]);
        if (i==n-1||!c[i]) {
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
            subopt += abs(temp_loss);
            for (int j = pre_i; j <= i; j++){
                gradient[j] = temp_loss;
            }
            pre_i = i + 1;
            temp_loss = 0;
        }
    }
    delete[] gsign;
    return subopt;
}




int train_tvaclm(const vector< vector<double>> x, vector< vector<double>>& f,const  vector<int> y , int q, const double lam, double *b, double L, const double M) {
    int n = y.size();
    int d = x.size();
    double duration = 0;
    vector<double> fsum(n, 0);
    vector<double> fsumtmp(n, 0);
    VectorXd db(q-1);
    vector<vector<int>> argsort(d, vector<int>(n));
    vector<vector<int>> argsort_c(d,vector<int>(n-1));
    vector<vector<int>> argsort_inv(d, vector<int>(n));
    set_argsort(argsort, argsort_c, argsort_inv, x);
    vector<vector<int>> sorted_y(d, vector<int>(n));
    for (int j = 0; j < d; j++){
        for (int i = 0; i < n; i++){
            // これは毎回やる必要はない
            sorted_y[j][i] = y[argsort[j][i]];
        }
    }
    vector<double> solver_y(n);
    vector<double> solver_f(n);
    vector<double> sorted_solver_y(n);
    vector<double> sorted_fsum(n);
    vector<double> sorted_f(n);
    vector<vector<double>> gradient(d, vector<double>(n));
    // vector<vector<double>> hessianb(q - 1, vector<double>(q - 1));
    MatrixXd hessianb(q-1,q-1);
    LLT<MatrixXd> llt;
    VectorXd newton(q-1);
    double newton_eta;
    double loss;
    double bsubopt, pre_bsubopt;
    double l_inv = 1/(L);
    double flsa_lam = l_inv * lam;
    double subopt;
    clm_b_derivative(n, q, &fsum[0], b, &y[0], M, db, hessianb);
    subopt = 0;
    for (int l = 0; l<q-1; l++){
        subopt += abs(db[l]);
    }
    // cout <<"b subopt: "<<subopt<< ", total_subopt: ";
    for (int j = 0; j < d; j++){
        for (int i = 0; i < n; i++){
            sorted_fsum[i] = fsum[argsort[j][i]];
            sorted_f[i] = f[j][argsort[j][i]];
        }
        subopt += calc_suboptibality(q, n, b, &sorted_fsum[0], &sorted_f[0],lam, argsort_c[j], &sorted_y[j][0], &gradient[j][0]);
    }

    // cout <<subopt << "\n";

    const double init_subopt = subopt;

    for (int k=0; k<ITER; k++) {
        chrono::system_clock::time_point  start, end; // 型は auto で可
        start = chrono::system_clock::now();
        for (int j = 0; j < d; j++){
            solve_square(n, q, &fsum[0], &f[j][0], b, &y[0], &solver_y[0], k, l_inv);
            for (int i = 0; i < n; i++){
                fsumtmp[i] = fsum[i] - f[j][i];
                sorted_solver_y[i] = solver_y[argsort[j][i]];
            }
            tf_dp(n, &sorted_solver_y[0], flsa_lam, &argsort_c[j][0], &solver_f[0]);
            for (int i = 0; i < n; i++){
                f[j][i] = solver_f[argsort_inv[j][i]];
                fsum[i] = fsumtmp[i] + f[j][i];
            }
        }
        end = chrono::system_clock::now();
        duration += chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); 
        subopt = 0;
        for (int j = 0; j < d; j++){
        // for (int j = d-1; j >= 0; j--){
            for (int i = 0; i < n; i++){
                sorted_fsum[i] = fsum[argsort[j][i]];
                sorted_f[i] = f[j][argsort[j][i]];
            }
            subopt += calc_suboptibality(q, n, b, &sorted_fsum[0], &sorted_f[0],lam, argsort_c[j], &sorted_y[j][0], &gradient[j][0]);
        }
        // if (k%pn==0){
        //     loss = calc_objective(fsum, y, b, q, f, argsort, lam);
        //     std::cout << "iter: " << k << " time: " << duration;
        //     std::cout <<"  loss: " << loss <<" , subopt" << " : "<<subopt/init_subopt <<"\n";
        // }
        
        if (subopt/init_subopt < PRE_FIT_EPS) {
            // std::cout << "pre-converged: " << k << "\n";
            // std::cout << duration << "\n";
            break;
        }
    }
    for (int k=0; k<ITER; k++) {
        chrono::system_clock::time_point  start, end; // 型は auto で可
        start = chrono::system_clock::now();
        clm_b_derivative(n, q, &fsum[0], b, &y[0], M, db, hessianb);
        llt.compute(hessianb);
        newton = llt.solve(db);
        
        for (int l = 0; l<q-1; l++) b[l] -= newton[l];
        for (int j = 0; j < d; j++){
        // for (int j = d-1; j >= 0; j--){
            solve_square(n, q, &fsum[0], &f[j][0], b, &y[0], &solver_y[0], k, l_inv);
            for (int i = 0; i < n; i++){
                fsumtmp[i] = fsum[i] - f[j][i];
                sorted_solver_y[i] = solver_y[argsort[j][i]];
            }
            tf_dp(n, &sorted_solver_y[0], flsa_lam, &argsort_c[j][0], &solver_f[0]);
            for (int i = 0; i < n; i++){
                f[j][i] = solver_f[argsort_inv[j][i]];
                fsum[i] = fsumtmp[i] + f[j][i];
            }
        }

        end = chrono::system_clock::now();
        duration += chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); 
        if (k%pm==0){

            if (b[0] < -M || b[q-2] > M) return 1;
            for (int l = 0; l<q-2; l++) if (b[l+1] - b[l] < 0) return 1;
            bsubopt = 0;
            subopt = 0;
            for (int l = 0; l<q-1; l++){
                bsubopt += abs(db[l]);
            }
            for (int j = 0; j < d; j++){
                for (int i = 0; i < n; i++){
                    sorted_fsum[i] = fsum[argsort[j][i]];
                    sorted_f[i] = f[j][argsort[j][i]];
                }
                subopt += calc_suboptibality(q, n, b, &sorted_fsum[0], &sorted_f[0],lam, argsort_c[j], &sorted_y[j][0], &gradient[j][0]);
            }

            if ((subopt + bsubopt)/init_subopt < EPS) {
                // std::cout << "converged: " << k << "\n";
                // std::cout << duration << "\n";
                // loss = calc_objective(fsum, y, b, q, f, argsort, lam);
                // std::cout << "iter: " << k << " loss: " << loss << endl;
                // std::cout << "f max: [";
                // for (int j=0; j<d; j++) std::cout << *std::max_element(f[j].begin(), f[j].end()) << ' ';
                // std::cout << "]" <<endl;

                // std::cout << "f min: [";
                // for (int j=0; j<d; j++) std::cout << *std::min_element(f[j].begin(), f[j].end()) << ' ';
                // std::cout << "]" <<endl;

                break;
            }
        }


        // if (k%pn==0){
        //     loss = calc_objective(fsum, y, b, q, f, argsort, lam);
        //     std::cout << "iter: " << k << " time: " << duration;
        //     std::cout <<"  loss: " << loss <<" , subopt" << " : "<< subopt/init_subopt << " , bsubopt" << " : "<<bsubopt/init_subopt << "\n";
        //     std::cout << "b: [";
        //     for (int l = 0; l<q-1; l++){
        //         std::cout << b[l] << " ";
        //     }
        //     std::cout << "]\n";
                    
        // }
    }
    return 0;
}




double solve_gradient_descent(const vector< vector<double>> x, vector< vector<double>>& f, const vector<int> y , int q, double lam, double *b, const double L, const double t0, const double M) {
    double lastloss;
    int n = y.size();
    int d = x.size();
    double subopt;
    vector<vector<double>> updated_f(d, vector<double>(n));
    vector<double> fsum(n, 0);
    vector<double> fsumtmp(n, 0);
    vector<vector<int>> argsort(d, vector<int>(n));
    vector<vector<int>> argsort_c(d,vector<int>(n));
    MatrixXd hessianb(q-1,q-1);
    LLT<MatrixXd> llt;
    VectorXd newton(q-1);
    VectorXd db(q-1);
    vector<vector<int>> argsort_inv(d, vector<int>(n));
    set_argsort(argsort, argsort_c, argsort_inv, x);
    vector<double> solver_y(n);
    vector<double> solver_f(n);
    double tmp_f;
    const double steepest_linv = sqrt(t0)/L;
    // cout << steepest_linv << "||||\n";
    double loss;

    vector<int> sorted_y(n);
    vector<double> sorted_fsum(n);
    vector<double> sorted_f(n);
    vector<double> sorted_solver_y(n);
    double duration = 0;

   
    vector<vector<double>> gradient(d, vector<double>(n));
    for (int k=0; k<ITER; k++) {
        clm_b_derivative(n, q, &fsum[0], b, &y[0], M, db, hessianb);
        llt.compute(hessianb);
        newton = llt.solve(db);
        
        for (int l = 0; l<q-1; l++) b[l] -= newton[l];
        if (b[0] < -M || b[q-2] > M) return 1;
        subopt= 0;

        chrono::system_clock::time_point  start, end; // 型は auto で可
        start = chrono::system_clock::now();
        for (int i = 0; i < n; i++){

            fsum[i] = 0;
            for (int j = 0; j < d; j++){
                fsum[i] += f[j][i];
            }
        }
        for (int j = 0; j < d; j++){
            for (int i = 0; i < n; i++){
                sorted_y[i] = y[argsort[j][i]];
                sorted_fsum[i] = fsum[argsort[j][i]];
                sorted_f[i] = f[j][argsort[j][i]];
            }
            subopt += calc_suboptibality(q, n, b, &sorted_fsum[0], &sorted_f[0],lam, argsort_c[j], &sorted_y[0], &gradient[j][0]);
        }

        end = chrono::system_clock::now();
        duration += chrono::duration_cast<std::chrono::microseconds>(end-start).count(); 

        
        if (k%pn==0){
            loss = calc_objective(fsum, y, b, q, f, argsort, lam);
            if (loss==INFINITY){
                return loss;
            }
        }
        
        if (subopt < EPS) {
            std::cout << "converged: " << k << "\n";
            break;
        }

        for (int j = 0; j < d; j++){
            for (int i = 0; i < n; i++){
                f[j][i] -= steepest_linv / sqrt(t0 + k) * gradient[j][argsort_inv[j][i]];
            }
        }
        // cout << "loss: " << loss;
        // cout << " . subopt" << " : "<<subopt << "\n";
    }
    return loss;
}





// PYBIND11_PLUGIN(flsaclm) {
//     py::module m("flsaclm", "flsaclm made by pybind11");
//     m.def("train_tvaclm", &train_tvaclm);
//     return m.ptr();
// }