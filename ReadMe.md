# Torch DE

Torch_DE is a framework for solving differential equations (DE) using neural networks via automatic differentiation (AD) or backprop. Fully compatable with Pytorch and Pytorch Lightning. 

Heavily inspired by PINN frameworks like Modulus, Torch_DE is designed to make the backend flexible and simple. 

Torch_DE's approach means a lot more pytorch boilerplate is placed on the user, however this is by design. While PINN frameworks like Modulus and DeepXDE are very powerful, working like magic and reducing boilerplate, it can be difficult to understand the backend and make changes to the training loop

Torch_DE's approach is to provide certain features and objects that can be inserted into a standard pytorch training loop allowing far more control in training process while reducing the more brittle and annoying parts of training PINNs (such as getting derivatives and sampling points from 2D geometry).


## Components
### Geometry
### Continuous
At the heart of this project is the DE_getter object which turns extracting derivatives from networks from a indexing task, which is brittle and error prone, into using strings to access these values:

```python
import torch
import torch.nn as nn
from torch_DE.continuous import DE_Getter

# Solving the Spring Equation u_tt = -u with u(0) = 0 and u_t(0) = 1

net = nn.Sequential(nn.Linear(1,200),nn.Tanh(),nn.Linear(200,1))
PINN = DE_Getter(net = net)
PINN.set_vars(input_vars= ['t'], output_vars= ['u'])
PINN.set_derivatives(derivatives=['u_t','u_tt'])

t = torch.linspace(0,2*torch.pi,100)

optimizer = torch.optim.Adam(params = net.parameters(), lr = 1e-3)
# Training Loop
for epoch in range(5000):
    #Calculate Derivatives
    out = PINN.calculate(t)
    #Spring Equation is u_tt + u = 0. Notice we can easily call derivatives and outputs using dicts rather than indexing
    residual = (out['u_tt'] + out['u']).pow(2).mean()

    #Data Fitting term
    data = out['u'][0].pow(2).mean() + (out['u_t'][0] - 1).pow(2).mean()

    loss = data + residual
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

```


Torch_DE uses functorch module and vmap to extract derivatives so currently multi-gpu and models with buffers (such as batch norm) may not work (This hasn't been tested and is a TO DO)

### Discrete
TBH
### Visualization
TBH


## To Do:
- Packaging
- Weighting methods
- Deep Operator Networks
- Discrete (Graph and Conv networks)
- Visualization Tools
- 3D Geometry (No way Im making a 3D Domain object)