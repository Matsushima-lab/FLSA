# FLSA
Generic implementation of the Fused Lasso signal operator 

# Mathematical discription
Input: 

$$y = (y_1,\ldots,y_n) \in \mathbb{R}^n, \ell: \mathbb{R} \times \mathbb{R} \to \mathbb{R}^+ , \lambda \in \mathbb{R}^+$$ 

Output: Solution of

$$\mathop{\mathrm{argmin}}\limits_{x= (x_1,\ldots,x_n) \in \mathbb{R}^n} \sum_{i=1}^n \ell(x_i,y_i) + \lambda \sum_{i=1}^{n-1}|x_{i}-x_{i+1}|$$
