# FLSA
Generic implementation of the Fused Lasso signal operator 

# Mathematical discription
Input: 

$$y = (y_1,\ldots,y_n) \in \mathbb{R}^n, \ell: \mathbb{R} \times \mathbb{R} \to \mathbb{R}^+ , \lambda \in \mathbb{R}^+$$ 

Output: Solution of

$$x^* = \mathop{\mathrm{argmin}}\limits_{x= (x_1,\ldots,x_n) \in \mathbb{R}^n} \sum_{i=1}^n \ell(x_i,y_i) + \lambda \sum_{i=1}^{n-1}|x_{i}-x_{i+1}|$$

in $\mathbb{R}^n$
# Algorithm
The algorithm is based on dynamic programming and consists of forawd and backward step

## Forward step
As an initialization, we set 

$$ \delta_1(x) = \ell(x_1,y_1) $$

From $i=2,\ldots ,n$, we compute 

$$\delta_i (x) = \min_{x_{i-1}} \delta_{i-1}(x_{i-1}) + \ell(x,y_i) + \lambda |x-x_{i-1}|$$

Then, $\mathop{\mathrm{argmin}}_{x} \delta_n (x)$ is $n$-th element of a solution of the problem.

## Backword step

As an initialization we get

$$x_n^* = \mathop{\mathrm{argmin}}_{x}\  \delta_n (x)$$

From $i=n-1,\ldots,1$, we compute

$$x_i^* = \mathop{\mathrm{argmin}}\limits_{x} \ \delta(x) + \lambda |x - x_{i+1}^*| $$
