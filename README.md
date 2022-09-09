# FLSA
Generic implementation of the Fused Lasso signal operator 

# Mathematical description
Input: 

$$y = (y_0,\ldots,y_{n-1}) \in \mathbb{R}^n, \ell: \mathbb{R} \times \mathbb{R} \to \mathbb{R}^+ , \lambda \in \mathbb{R}^+$$ 

Output: Solution of fused lasso problem $x^*\in\mathbb{R}^n$ defined as 

$$x^*=\mathop{\mathrm{argmin}}\limits_{x= (x_0,\ldots,x_{n-1}) \in \mathbb{R}^n}\ \sum_{i=0}^{n-1} \ell(x_i,y_i) + \lambda \sum_{i=0}^{n-2}|x_{i}-x_{i+1}|.$$

# Implementation

Class for function $\delta$ has to be implemented for each loss.
Currently, a class for squared loss $\ell(x,y) = (x-y)^2$ and logistic loss $\ell(x,y) = -\log(1+\exp(-yx))$ are going to be implemented.

# Usage
Example:
```
>>> import flsa_dp
>>> y = [0,1]
>>> λ = 0.5
>>> x = flsa_dp.solve(y, λ, "squared")
[0.5, 0.5]
```

# Algorithm
The algorithm is based on dynamic programming and consists of forawd and backward step. 
The key for the efficiency is to deal with $\delta_i:\mathbb{R}\to\mathbb{R}$ for $i=1,\ldots,n$ defined in below.
Usually it is done in $O(n)$, which is very efficient.

## Forward step
As an initialization, we set 

$$ \delta_0(x) = \ell(x,y_0) $$

From $i=1,\ldots ,n-1$, we compute 

$$\delta_i (x) = \min_{x_{i-1}\in \mathbb{R}}\ \delta_{i-1}(x_{i-1}) + \ell(x,y_i) + \lambda |x-x_{i-1}|$$

Then, $\mathop{\mathrm{argmin}}_{x} \delta_{n-1} (x)$ is $n$-th element of a solution of the problem.

## Backword step

As an initialization we get

$$x_{n-1}^* = \mathop{\mathrm{argmin}}_{x\in \mathbb{R}}\  \delta_{n-1} (x)$$

From $i=n-2,\ldots,0$, we compute

$$x_i^* = \mathop{\mathrm{argmin}}\limits_{x\in \mathbb{R}} \ \delta(x) + \lambda |x - x_{i+1}^*| $$
