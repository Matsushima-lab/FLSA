# implementation Deatils
We consider loss functions such that $\delta_i$ is represented as 

$$ \delta_i(x) = \sum_{t=1}^{N} g_t(x)  \mathbb{I}[k_t < x < k_{t+1}]. $$

and $\delta_i$ is continuoudly differentiable. Here, $\mathbb{I}[\bullet]$ is the indicator function.
$-\infty = k_1 < \cdots < k_{N+1} = \infty$ are called knots.

Abstract `class DeltaFunc` has
```python
knots
knots_n
```
for $(k_t)$ and $N$ as instance variables and 

```python  
def forward(self, y):  # return next delta for a given delta
def backward(self, b): # return previous b for a given b
def find_min(self): # find a root of delta(x) = 0
```  
for virtual methods.

Information on $g_t$s are stored in cocrete classes. 
To interact, we need

```python
def calc_derivative_at(t):
```
This method computes $\delta'(k_t)$ for given $t$ 


```python
def calc_inverse(t,d):   
```
This method finds a root of $g_t'(x)=d$

for `find_min(self) `.
To update the infomation on $g$

```python
def add_linear(b):   
```
This method updates $g_t(x) \gets g_t(x) + bx$ for each $t$

```python
def overwrite_right(y,k):   
```
This method finds $t'$ such that $k_t < k < k_{t+1}$ and update 
$$ \delta(x) = \sum_{t=1}^{N} g_t(x)  \mathbb{I}[k_t < x < k_{t+1}]. $$

by $k_{t'+1} \gets k, k_{t'+2} \gets +\infty, g_{t'+1}(x) \gets \ell(x,y), and  $N \gets t'+1$.

```python
def overwrite_left(y,k):   
```
This method finds $t'$ such that $k_t < k < k_{t+1}$ and update 

$$ \delta(x) = \sum_{t=1}^{N} g_t(x)  \mathbb{I}[k_t < x < k_{t+1}]. $$

by $k_1 \gets k, g_1(x) \gets \ell(x,y)$ and $k_t \gets k_{t-t'+1},  g_t(x) \gets g_{t-t'+1}(x)$  for $t =1,\ldots, N$ in which $N \gets N - t' +1$.


## case of squared loss

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
for $(a_t),(b_t)$
### Forward step
1. Initialize $\delta_1(x) = \ell(x,y_1) = (x-y_1)^2$
2. For $t=2,\ldots,n$,
```python
knots = [bm] + knots[i:j] + [bp] 
coef_a = [coef_am] + coef_a[i:j] + [coef_ap]
coef_b = [coef_bm] + coef_b[i:j] + [coef_bp]
```
Here, .......

### Find_min step
1. Find $t'$ such that $\delta_n(k_{t'}) \le 0 < \delta_n(k_{t'+1})$, i.e., `derivative_at(t)` returns non-negative value 
2. Find $x$ such that $g'_{t'}(x) = 0$   

### Backward step



