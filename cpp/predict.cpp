#include "flsa_dp.hpp"
#include "predict.hpp"
#include "solve_block_coordinate_descent.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <string>
#include <fstream>
#include <stdexcept>


using namespace std;

void calc_metrix(const vector<vector<double>>& sortedf, const vector<vector<double>>& sortedx, const vector<vector<double>>& valx, const vector<int>& valy, double& mae, double& acc, double& prob, const int q, const double *b, const double M){
    mae = 0;
    acc = 0;
    prob = 0;
    
    double qprobmax, f_train, pred, qprob;
    const int d = sortedx.size();
    const int n = sortedx[0].size();

    const int valn = valy.size();
    for (int k = 0; k < valn; k++){
        qprobmax = 0;
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
        for (int l = 1; l<=q; l++){
            if (l==1) qprob = sigmoid(b[0] - f_train) - sigmoid(- f_train);
            else if (l<q) qprob = sigmoid(b[l - 1] - f_train) - sigmoid(b[l - 2]- f_train);
            else qprob = sigmoid(M - f_train) - sigmoid(b[q - 2]- f_train);
            if (l == valy[k]) prob += qprob/(sigmoid(M - f_train) - sigmoid(-M- f_train));
            if (qprobmax < qprob){
                qprobmax = qprob;
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
