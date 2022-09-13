# Implementation Details
Abstract `class DeltaFunc` has

```python  
class DeltaFunc:
    def forward(self, y):  # return next delta for a given delta
    def backward(self, b): # return previous b for a given b
    def find_tangency(self, s): # find argmin_x delta(x) + sx
```  
for virtual methods.

Then `solve` can be written as follows:
```python  
def solve(y: np.array, lamb: float, loss: str = None) -> np.array:
    n = y.size
    x = [np.nan] * n
    delta = [None] * n

    # Forward Step    
    delta[0] = DeltaFunc(loss, y[0]) # delta_0(x) = ell(x,y_0)
    for i in range(n - 1):
        delta[i + 1] = delta[i].forward(lamb, y[i], y[i + 1])

    # Backward Step
    x[n - 1] = delta[n - 1].find_tangency(0)     # Find_min Step
    for i in range(n - 1, 0, -1):
        x[i - 1] = delta[i].backward(x[i])

    return x
```  

For differnt losses, we need to create different concrete classes that inherit DeltaFunc:
```python
class DeltaFunc:
    def __init__(self, loss, y0):
        if loss == 'squared': 
            return DeltaSquared(y[0]) 
        elif loss == 'logistic':
            return DeltaLogistic(y[0]) 
        else:
            raise RuntimeError('Invalid Loss Name!!!')

```

Concrete classes have to have the following methods implemented:

```python

class DeltaFunc:
    @abstractmethod
    def __add_loss(y):
    # Update delta(x) <- delta(x) + loss(x,y)
        pass
    @abstractmethod        
    def __overwrite(bm,bp,λ):
    # Update delta(x) <- [[x < bm]] delta(bm) -λx +  [[bm <= x <= bp]] delta(x) + [[bp < x ]] delta(bp) + λx  
        pass
    @abstractmethod
    def __find_tangency(s):
    # Return x that minimizes delta(x) - sx
```

# Implementation Deatils for DeltaFunc

We consider loss functions such that $\delta_i$ is represented as 

$$ \delta_i(x) = \sum_{t=0}^{N-1} g_t(x)  \mathbb{I}[k_t < x < k_{t+1}]. $$

and $\delta_i$ is continuoudly differentiable. Here, $\mathbb{I}[\bullet]$ is the indicator function.
$-\infty = k_0 < \cdots < k_{N} = \infty$ are called knots. $N$ is a number of knots.
has (<- it should also be stored in concrete classes..)

```python
knots
knots_n
```
for $(k_t)$ and $N$ as instance variables and 
Information on $g_t$ s are stored in cocrete classes. 


```python
def __add_loss(t, y):   
```
This method updates $g_t(x) \gets g_t(x) + \ell(x,y)$ for all $t$.

```python
def __overwrite(bm,bp,λ):   
```
This method finds $t',t''$ such that $k_{t'} < b^- < k_{t'+1}$ and $k_{t''} < b^+ < k_{t''+1}$ update 

$$ \delta(x) = \sum_{t=1}^{N} g_t(x)  \mathbb{I}[k_t < x < k_{t+1}]. $$

by

$$ g_1(x ) = g(b^-) -\lambda x,\ g_t(x) \gets g_{t - t' + 1}(x),\  g_{t'' + 1}(x) \gets g(b^+) +\lambda x, \ N\gets t'' -t' +3  $$



## Case of squared loss

For `class DeltaSquared(DeltaFunc)` we see $\delta_1(x) = \ell(x,y_1) = (x-y_1)^2$ and 

$$\delta_2(x) = \min_{x_1}\ (x_1-y_1)^2 + \lambda |x -x_1|, $$

which is a form of 

$$ \sum_{t=1}^{N} (a_t x^2 + b_t x + c_t )  [k_t < x < k_{t+1}]. $$

It turns out $\delta_i(x)$ in general can be written in this form in squared loss case.
Thus, we store knots $(k_t)$, number of knots $N$, and coefficients $(a_t),(b_t),(c_t)$. It turns out $(c_t)$ is unnessesary.
```python
class DeltaSquared(DeltaFunc):
  def __init__(self,y):
    self.knots_n = 1
    self.knots = [-float('inf'), float('inf')] 
    self.coef_a = [1]
    self.coef_b = [-2 * y]
```
`self.coef_a` and `self.coef_b` are for $(a_t),(b_t)$.

Since $(a_t x^2 + b_t x + c_t)' =2a_t x+ b_t,$ `__calc_derivative_at(t)` can be implemented by $O(1)$.
```python  
def __calc_derivative_at(t):
    return 2*self.coef_a[t] * self.knots[i] + self.coef_b[t]  
```

Since $(a_t x^2 + b_t x + c_t)' =d \Leftrightarrow x = (d - b_t)/2a_t,$ `__calc_inverse`can be implemented by $O(1)$.
```python  
def __calc_inverse(t):
    return (d-self.coef_b[t])/2*self.coef_a[t]   
```

Since $(a_t x^2 + b_t x + c_t) + (x-y)^2 = (a_t+1) x^2 + (b_t-2y)x + \mathrm{const.},$
```python  
def __add_loss(t,y):
    self.coef_a[t] += 1
    self.coef_b[t] += -2*y
```









