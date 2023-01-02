#include <vector>
#include <eigen3/Eigen/Dense>
using namespace std;
using namespace Eigen;

double sigmoid(const double x);
double clm_w_derivative(const int q, const double s, const double *b, const int y, const double M);
void clm_b_derivative(int n, int q, const double *f, const double *b, const int *y, const double M, VectorXd& db, MatrixXd& hessianb);
double log1pexpz(const double z);
double log_likelihood(const double fi, const int yi, const double *b, const int q, const double M);
int trainClm(const vector< vector<double>> x, const vector<int> y, const int q, vector<double> &w ,double *b, const double L, const double M);
