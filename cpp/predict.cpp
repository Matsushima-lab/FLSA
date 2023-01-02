#include "flsa_dp.hpp"
#include "predict.hpp"
#include "train_tvaclm.hpp"
#include "clm.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <string>
#include <fstream>
#include <stdexcept>


const double LOG2 = 0.693147;
using namespace std;

double total_logprob(const double fi, const double M){
    double delta = 2 * M;
    double gamma = -M - fi;
    double c1;
    if (delta < LOG2) c1 = log(-expm1(-delta));
    else c1 = log1p(-exp(-delta));
    return  log1pexpz(-gamma-delta) + log1pexpz(gamma) - c1;
}
    


void tvaclm_calc_metrix(const vector<vector<double>>& sortedf, const vector<vector<double>>& sortedx, const vector<vector<double>>& valx, const vector<int>& valy, double& mae, double& acc, double& prob, const int q, const double *b, const double M){
    mae = 0;
    acc = 0;
    prob = 0;
    
    double qprobmax, f_train, pred, qprob, logqprob, tlp;
    const int d = sortedx.size();
    const int n = sortedx[0].size();

    const int valn = valy.size();
    for (int k = 0; k < valn; k++){
        qprobmax = 1e100;
        f_train = 0;
        pred = 0;

        for (int j = 0; j< d; j++){
            if (sortedx[j][n-1] < valx[j][k]) f_train+=sortedf[j][n-1];
            else{
                for (int i = 0; i< n;i++){
                    if (sortedx[j][i] >= valx[j][k]) 
                    {
                        f_train+=sortedf[j][i];
                        break;
                    }
                }
            }
        }

        tlp = total_logprob(f_train, M);
        for (int l = 1; l<=q; l++){
            logqprob = log_likelihood(f_train, l, b,q, M);
            if (l == valy[k]) prob += exp(-logqprob - tlp);
            if (qprobmax > logqprob){
                qprobmax = logqprob;
                pred = l;
            }
        }
        if (pred==valy[k]) acc+=1;
        mae+=abs(pred-valy[k]);

    // std::cout << k<<" : " <<tlp << " : "<<pred<<" : "<<valy[k] <<endl;
    }

    prob/=valn;
    acc/=valn;
    mae/=valn;

    // std::cout << "PROB: " << prob <<"| ";
    // std::cout << "ACC: " << acc <<"| ";
    // std::cout << "MAE: " << mae <<"\n";
}

void clm_calc_metrix(const vector<double> w, const vector<vector<double>> valx, const vector<int> valy, double& mae, double& acc, double& prob, const int q, const double *b, const double M){
    mae = 0;
    acc = 0;
    prob = 0;
    
    double qprobmax, f_train, pred, qprob, logqprob, tlp;
    const int d = valx.size();
    const int valn = valy.size();
    for (int k = 0; k < valn; k++){
        qprobmax = 0;
        f_train = 0;
        pred = 0;

        for (int j = 0; j<d; j++){
            f_train+=w[j] * valx[j][k];
        }
        f_train += w[d];
        tlp = total_logprob(f_train, M);
        for (int l = 1; l<=q; l++){
            logqprob = log_likelihood(f_train, l, b,q,M);
            if (l == valy[k]) prob += exp(tlp - logqprob);
            if (qprobmax < logqprob){
                qprobmax = logqprob;
                pred = l;
            }
        }
        if (pred==valy[k]) acc+=1;
        mae+=abs(pred-valy[k]);
    }

    prob/=valn;
    acc/=valn;
    mae/=valn;

    // std::cout << "PROB: " << prob <<"| ";
    // std::cout << "ACC: " << acc <<"| ";
    // std::cout << "MAE: " << mae <<"\n";
}