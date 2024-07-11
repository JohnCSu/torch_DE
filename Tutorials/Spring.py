import torch
import torch.nn as nn
from torch_DE.continuous import DE_Getter
from torch_DE.utils.data import PINN_dataset,PINN_Dataloader
from torch_DE.utils import Loss_handler
from matplotlib import pyplot as plt
'''
Solving the Spring Equation 

u_tt = -u with u(0) = 0 and u_t(0) = 1

With Analytic solution u=sin(x).

This demonstrates the basic workflow of torch_DE. You will notice if you go through other examples, the workflow is almost identical, 
even if we go to complicated fluid flow problems. The only difference with the other examples is we bring out some additional features
that help train PINNs.

'''
#Torch DE assumes first dimension is the batch dimension
t = torch.linspace(0,2*torch.pi,10_000).unsqueeze(dim =-1)
t0 = torch.tensor([0.]).unsqueeze(dim=-1)

dataset = PINN_dataset()
dataset.add_group('collocation_points',t,batch_size=1000,shuffle=True)
dataset.add_group('initial condition',t0,batch_size= 1)
DL = PINN_Dataloader(dataset)

#Losses
spring_eq = lambda u,u_tt,**kwargs: u_tt+u
losses = Loss_handler(dataset.group_names())
losses.add_initial_condition('initial condition',{'u': 0., 'u_t': 1.})
losses.add_residual('collocation_points',{'spring_eq':spring_eq})

#Usual Pytorch set up
net = nn.Sequential(nn.Linear(1,200),nn.Tanh(),nn.Linear(200,1))
optimizer = torch.optim.Adam(params = net.parameters(), lr = 1e-3)
net=net.cuda()
optimizer.zero_grad()

#Set PINN
PINN = DE_Getter(net = net)
PINN.set_vars(input_vars= ['t'], output_vars= ['u'])
PINN.set_derivatives(derivatives=['u_t','u_tt'])

# Training Loop
for epoch in range(1001):
    for x in DL:
        x = x.to(device = 'cuda')
        #Calculate Derivatives
        out = PINN.calculate(x)
        #Get Losses
        loss = losses(x,out)
        loss.sum().backward()

        optimizer.step()
        optimizer.zero_grad()
    if (epoch % 100) == 0:
        print(f'Epoch {epoch} Loss {float(loss.sum()):.3E}')

#Visulisation
with torch.no_grad():
    t = torch.linspace(0,2*torch.pi,100)
    u = net.cpu()(t.unsqueeze(-1))
    plt.plot(t,u,label = 'PINN Solution')
    plt.plot(t,torch.sin(t),label = 'Analytic',linestyle = '--')
    plt.title(f'PINN training after {epoch} Epochs')
    plt.legend()
    plt.show()