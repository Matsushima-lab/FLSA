# implementation Deatils

```
class DeltaFunc
```


## case of squared loss

### Forward step

We see $\delta_1(x) = \ell(x,y_1) = (x-y_1)^2 $. Then,

$$\delta_2(x) = \min_{x_1}\ (x_1-y_1)^2 + \lambda |x -x_1|, $$

which is a form of 

$$ \delta_2(x) = \sum_{t=1}^{N} (a_t x^2 + b_t x + c_t )  [k_t < x < k_{t+1}]. $$

It turns out $\delta_n$ can be written in this form in squared loss case in general.
Thus, we store knots $(k_t)$, number of knots $N$, and coefficients $(a_t),(b_t),(c_t)$. It turns out $(c_t)$ is unnessesary.

### Backward step
