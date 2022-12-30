#include <vector>

using namespace std;
void tvaclm_calc_metrix(const vector<vector<double>>& sortedf, const vector<vector<double>>& sortedx, const vector<vector<double>>& valx, const vector<int>& valy, double& mae, double& acc, double& prob, const int q, const double *b, const double M);
void clm_calc_metrix(const vector<double> w, const vector<vector<double>> valx, const vector<int> valy, double& mae, double& acc, double& prob, const int q, const double *b, const double M);