# FLSA
Generic implementation of the Fused Lasso signal operator 

# Mathematical discription
Input: $ y = (y_i)_i \in R^n, \ell: R \times R \to R_+ , \lambda \in R_+$ 

Output: Solution of


$$  \mathop{\mathrm{argmin}}_{x} \sum_{i=1}^n \ell(x_i,y_i) + \lambda \sum_{i=1}^{n-1}|x_{i}-x_{i+1}|$$
