# %%
import torch
from welford_torch import Welford

# Initialize Welford object
w = Welford()

# Itorchut data samples sequentialy
w.add(torch.tensor([0, 100]))
w.add(torch.tensor([1, 110]))
w.add(torch.tensor([2, 120]))

# output
print(w.mean)  # mean --> [  1. 110.]
print(w.var_s)  # sample variance --> [1, 100]
print(w.var_p)  # population variance --> [ 0.6667 66.6667 ]

# You can add other samples after calculating variances.
w.add(torch.tensor([3, 130]))
w.add(torch.tensor([4, 140]))

# output with added samples
print(w.mean)  # mean --> [  2. 120.]
print(w.var_s)  # sample variance --> [  2.5 250. ]
print(w.var_p)  # population variance --> [  2. 200.]
# %%
import torch
from welford_torch import Welford

# Initialize Welford object with samples.
ini = torch.tensor([[0, 100], [1, 110], [2, 120]])
w = Welford(ini)

# output
print(w.mean)  # mean --> [  1. 110.]
print(w.var_s)  # sample variance --> [1, 100]
print(w.var_p)  # population variance --> [ 0.6667 66.6667]

# add other samples through batch method
other_samples = torch.tensor([[3, 130], [4, 140]])
w.add_all(other_samples)

# output with added samples
print(w.mean)  # mean --> [  2. 120.]
print(w.var_s)  # sample variance --> [  2.5 250. ]
print(w.var_p)  # population variance --> [  2. 200.]


# %%
import torch
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
print(w_1.var_p)  # population variance -->[ 0.6667 66.6667]

# On w_2
w_2.add(torch.tensor([3, 130]))
w_2.add(torch.tensor([4, 140]))
print(w_2.var_s)  # sample variance -->[ 0.5 50. ]
print(w_2.var_p)  # sample variance -->[ 0.25 25.  ]

# You can Merge objects to get variance of WHOLE samples
w_1.merge(w_2)
print(w.var_s)  # sample variance --> [  2.5 250. ]
print(w_1.var_p)  # sample variance -->[  2. 200.]

# %%
