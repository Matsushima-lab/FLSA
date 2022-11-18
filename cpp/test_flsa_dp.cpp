#include "flsa_dp.hpp"
#include <iostream>
using namespace std;

int main(){
    int n = 3;
    double y[] = {0,1,1};
    int c[] = {1,0};
    double beta[n];
    double lam = 0.01;
    tf_dp(n, y,lam, c, beta);
    for (int i=0 ; i<n ; i++ )
    {
    cout << beta[i]<<"\t";
    }
    return 0;
}