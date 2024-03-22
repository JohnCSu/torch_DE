import torch
import torch.nn as nn
from functorch import jacrev,jacfwd,vmap,make_functional
from torch_DE.continuous import DE_Getter


# Collocation Points (From 0 to 2pi)
t_col = torch.rand((998,1))*2*torch.pi

# Initial conditions u(0) = 0 , u_t(0) = 1
t_data = torch.tensor([0]).unsqueeze(-1)

t = torch.cat([t_data,t_col])

print(t.shape)
net = nn.Sequential(nn.Linear(1,200),nn.Tanh(),nn.Linear(200,1))
# Spring Equation

PINN = DE_Getter(net = net)
PINN.set_vars(input_vars= ['t'], output_vars= ['u'])
PINN.set_derivatives(derivatives=['u_t','u_tt'])
PINN.set_deriv_method('AD')
optimizer = torch.optim.Adam(params = net.parameters(), lr = 1e-3)

# For Loop
for i in range(1):
    output = PINN.calculate(t)
    
    print(PINN.derivatives)
    out = output['all']
    #Spring Equation is u_tt + u = 0. Notice we can easily call derivatives and outputs by strings rather than having to do
    #indexing
    residual = (out['u_tt'] + out['u']).pow(2).mean()

    #Data Fitting. In this case we know that the first element is the point t=0
    data = out['u'][0].pow(2).mean() + (out['u_t'][0] - 1).pow(2).mean()


    loss = data + residual
    print(f'Epoch {i} Total Loss{float(loss)}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

