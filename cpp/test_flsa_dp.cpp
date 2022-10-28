#include "flsa_dp.hpp"
#include <iostream>
using namespace std;

int main(){
    int n = 4;
    double y[] = {1,2,3,1};
    bool c[] = {0,0,1};
    double beta[n];
    double lam = 0.1;
    tf_dp(n, y,lam, c, beta);
    for (int i=0 ; i<n ; i++ )
    {
    cout << beta[i]<<"\t";
    }
    return 0;
}