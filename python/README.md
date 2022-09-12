# Implementation Details
Abstract `class DeltaFunc` has

```python  
class DeltaFunc:
    def forward(self, y):  # return next delta for a given delta
    def backward(self, b): # return previous b for a given b
    def find_min(self): # find a root of delta(x) = 0
```  
for virtual methods.

Then `solve` can be written as follows:
```python  
class DeltaFunc:
    def solve(y: np.array, lamb: float, loss: str = None) -> np.array:
        n = y.size
        x = [np.nan] * n
        delta = [None] * n

        # Forward Step    
        delta[0] = DeltaFunc(loss, y[0]) # delta_0(x) = ell(x,y_0)
        for i in range(n - 1):
            delta[i + 1] = delta[i].forward(lamb, y[i], y[i + 1])

        # Backward Step
        x[n - 1] = delta[n - 1].find_min()     # Find_min Step
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

# Implementation Deatils for Delta functions

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

### Find_min step
1. Find $t'$ such that $\delta_n(k_{t'}) \le 0 < \delta_n(k_{t'+1})$, i.e., `__derivative_at(t)` returns non-negative value 
2. Find $x$ such that $g'_{t'}(x) = 0$   


```python
def __calc_derivative_at(t):
```
This method computes $\delta'(k_t)$ for given $t$ 

```python
def __calc_inverse(t,d):   
```
This method finds a root of $g_t'(x)=d$

Then `find_min(self) ` can be written as follows:
```python
def find_min(self):
    for i in range(self.knots_n):
        if self.__calc_derivative_at(i) >= 0:
            break
    return self.__calc_inverse(i,0)
```
### Forward step
To update the infomation on $g$

```python
def __add_loss(t, y):   
```
This method updates $g_t(x) \gets g_t(x) + \ell(x,y)$.

```python
def __overwrite_right(y,k):   
```
This method finds $t'$ such that $k_t < k < k_{t+1}$ and update 
$$ \delta(x) = \sum_{t=1}^{N} g_t(x)  \mathbb{I}[k_t < x < k_{t+1}]. $$

by $k_{t'+1} \gets k, k_{t'+2} \gets +\infty, g_{t'+1}(x) \gets \ell(x,y), and  $N \gets t'+1$.

```python
def __overwrite_left(y,k):   
```
This method finds $t'$ such that $k_t < k < k_{t+1}$ and update 

$$ \delta(x) = \sum_{t=1}^{N} g_t(x)  \mathbb{I}[k_t < x < k_{t+1}]. $$

by $k_1 \gets k, g_1(x) \gets \ell(x,y)$ and $k_t \gets k_{t-t'+1},  g_t(x) \gets g_{t-t'+1}(x)$  for $t =1,\ldots, N$ in which $N \gets N - t' +1$.

```python
def forward(self, λ, y): #<- this should be wrong but roughly correct
    next = self.copy()
    for i in range(self.knots_n,0,-1):
        if self.__derivative_at(i) < λ:
            break
    bp = self.__calc_inverse(i,λ)
    next.__overwrite_right(y,i,bp)

    for i in range(self.knots_n):
        if self.__derivative_at(i) > -λ :
            break
    bm = self.__calc_inverse(i,-λ)
    next.__overwrite_left(y,i,bm)
    
    for i in range(self.knots_n):
        next.__add_loss(i,y) 
    return next
```

### Backward step

The solution of

$$ \mathop{\mathrm{argmin}}\ \delta_i(x) + \lambda | x-x_{i+1}|$$

is either $x_{i+1}$ or a point in which $\delta'_i(x) = \pm \lambda$.

```python
def backward(self, x): #<- this should be wrong. 
    for i in range(self.knots_n):
        if self.__calc_derivatice(i) < lambda:
            break
    bp = self.__calc_inverse(i,+lambda)
    for i in range(self.knots_n):
        if self.__calc_derivatice(i) < -lambda:
            break
    bm = self.__calc_inverse(i,-lambda)
    return max(min(x,bm),bp)
```


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









