# Welford
Python (Pytorch) implementation calculating Standard Deviation, Variance,
Covariance Matrix, and Whitening Matrix online and in parallel. This makes it
more memory efficient than computing in the standard way.

This implementation uses Welford's algorithm for variance, and standard
deviation. Online Covariance calculation uses a generalization shown on Wikipedia.

The algorithm is described in the followings,

* [Wikipedia:Welford Online Algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm)
* [Wikipedia:Welford Parallel Algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm)
* [Wikipedia:Covariance Online Algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online)

Welford's method is more numerically stable than the standard method. The theoretical background of Welford's method is mentioned in detail on the following blog articles. Please refer them if you are interested in.

* http://www.johndcook.com/blog/standard_deviation
* https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/

This library is a fork of the `welford` library implemented in Numpy ( https://github.com/a-mitani/welford ).

I later added the covariance calculation inspired by the implementation by Carsten Schelp (
https://carstenschelp.github.io/2019/05/12/Online_Covariance_Algorithm_002.html
). The `OnlineCovariance` class has feature parity with the normal `Welford` class,
but takes more memory and compute.

## Install
Download package via [PyPI repository](https://pypi.org/project/welford-torch/)
```
$ pip install welford-torch
```

## Example (OnlineCovariance)

Example showing how to use `OnlineCovariance` to compute the covariance and
other quantities. If only computing mean, standard deviation and variance, use
the `Welford` class as it is faster and uses less memory. Otherwise, both
classes have the same functionality
(i.e: `.add()`, `.add_all()` and `.merge()`. See below for examples).

```python
import torch
from welford_torch import OnlineCovariance

# Initialize Welford object, and add samples
w = OnlineCovariance()

dataset = torch.tensor([[-1, 90], [0, 100], [1, 110], [-1, 100], [1, 100]])
for datapoint in dataset:
    w.add(datapoint)

# output
print(w.mean)   # Mean --> [  0., 100.]
print(w.var_s)  # Sample variance --> [ 1., 50.]
print(w.var_p)  # Population variance --> [ 0.8000, 40.0000]
print(w.cov)      # Covariance matrix --> [[ 0.8000,  4.0000], [ 4.0000, 40.0000]]
print(w.corrcoef) # Pearson correlation coefficient --> [[1.0, 0.7071], [0.7071, 1.0000]]
print(w.eig_val)  # Eigenvalues (ascending) -> [ 0.3960, 40.4040]
print(w.eig_vec)  # Eigenvectors -> [[-0.9949,  0.1005], [ 0.1005,  0.9949]]
print(w.whit)     # Whitening Matrix [[ 1.5746, -0.1431], [-0.1431,  0.1718]]
print(w.whit_inv) # Whitening Matrix Inverse [[0.6871, 0.5726], [0.5726, 6.2986]]
```

## Example (Welford)
### For Online Calculation
```python
import numpy as torch
from welford_torch import Welford

# Initialize Welford object
w = Welford()

# Input data samples sequentialy
w.add(torch.tensor([0, 100]))
w.add(torch.tensor([1, 110]))
w.add(torch.tensor([2, 120]))

# output
print(w.mean)  # mean --> [  1. 110.]
print(w.var_s)  # sample variance --> [1, 100]
print(w.var_p)  # population variance --> [ 0.6666 66.66]

# You can add other samples after calculating variances.
w.add(torch.tensor([3, 130]))
w.add(torch.tensor([4, 140]))

# output with added samples
print(w.mean)  # mean --> [  2. 120.]
print(w.var_s)  # sample variance --> [  2.5 250. ]
print(w.var_p)  # population variance --> [  2. 200.]
```

Welford object supports initialization with data samples and batch addition of samples.
```python
# Initialize Welford object with samples
ini = torch.tensor([[0, 100],
                [1, 110],
                [2, 120]])
w = Welford(ini)

# output
print(w.mean)  # mean --> [  1. 110.]
print(w.var_s)  # sample variance --> [1, 100]
print(w.var_p)  # population variance --> [ 0.66666667 66.66666667]

# add other samples through batch method
other_samples = torch.tensor([[3, 130],
                          [4, 140]])
w.add_all(other_samples)

# output with added samples
print(w.mean)  # mean --> [  2. 120.]
print(w.var_s)  # sample variance --> [  2.5 250. ]
print(w.var_p)  # population variance --> [  2. 200.]
```

### For Parallel Calculation
Welford also offers parallel calculation method for variance.
```python
import numpy as torch
from welford_torch import Welford

# Initialize two Welford objects
w_1 = Welford()
w_2 = Welford()

# Each object will calculate variance of each samples in parallel.
# On w_1
w_1.add(torch.tensor([0, 100]))
w_1.add(torch.tensor([1, 110]))
w_1.add(torch.tensor([2, 120]))
print(w_1.var_s)  # sample variance -->[  1. 100.]
print(w_1.var_p)  # population variance -->[ 0.66666667 66.66666667]

# On w_2
w_2.add(torch.tensor([3, 130]))
w_2.add(torch.tensor([4, 140]))
print(w_2.var_s)  # sample variance -->[ 0.5 50. ]
print(w_2.var_p)  # sample variance -->[ 0.25 25.  ]

# You can Merge objects to get variance of WHOLE samples
w_1.merge(w_2)
print(w.var_s)  # sample variance --> [  2.5 250. ]
print(w_1.var_p)  # sample variance -->[  2. 200.]

```
