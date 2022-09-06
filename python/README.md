# implementation Deatils
We consider loss functions such that $\delta_i$ is represented as 

$$ \delta_i(x) = \sum_{t=1}^{N} g_t(x)  \mathbb{I}[k_t < x < k_{t+1}]. $$

and $\delta_i$ is continuoudly differentiable. Here, $\mathbb{I}[\bullet]$ is the indicator function.

Abstract DeltaFunc class has
```
class DeltaFunc
  
  def forward
  def backward
  def find_min
```  

For find_min we need

```
def calc_derivative_at(t)
```
This method computes $\delta'(k_t)$ for given $t$ 


```
def calc_root_of(t)   
```
This method finds a root of $g_t'(x)=0$

## case of squared loss

```
class DeltaSquared(DeltaFunc)
  
```


### Forward step

We see $\delta_1(x) = \ell(x,y_1) = (x-y_1)^2 $. Then,

$$\delta_2(x) = \min_{x_1}\ (x_1-y_1)^2 + \lambda |x -x_1|, $$

which is a form of 

$$ \delta_2(x) = \sum_{t=1}^{N} (a_t x^2 + b_t x + c_t )  [k_t < x < k_{t+1}]. $$

It turns out $\delta_n$ can be written in this form in squared loss case in general.
Thus, we store knots $(k_t)$, number of knots $N$, and coefficients $(a_t),(b_t),(c_t)$. It turns out $(c_t)$ is unnessesary.

### Backward step
