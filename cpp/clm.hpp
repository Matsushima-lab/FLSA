#include <vector>
#include <eigen3/Eigen/Dense>
using namespace std;
using namespace Eigen;

double sigmoid(const double x);
double clm_w_derivative(int q, double s, double *b, int y);
void clm_b_derivative(int n, int q, const double *f, const double *b, const int *y, const double M, VectorXd& db, MatrixXd& hessianb);
double log1pexpz(double z);
int trainClm(const vector< vector<double>> x, const vector<int> y, const int q, vector<double> &w ,double *b, const double L, const double M);
