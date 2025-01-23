from torch_DE.equations import get_NavierStokes,get_derivatives
from torch_DE.geometry.shapes import *
from torch_DE.continuous.Engines import FD_engine
from torch_DE.continuous import DE_Getter
from torch_DE.continuous.Networks import MLP,Wang_Net,Fourier_Net
from torch.optim.lr_scheduler import StepLR
from torch_DE.utils import Loss_handler,GradNorm
from torch_DE.utils.data import PINN_Dataloader,PINN_dataset
'''
The Non Linear Allen Cahn Equation: 

u_t - 0.0001*u_xx + 5*u^3 -5*u = 0

This Example highlights using the Causal Weighting Feature in torch DE. It is effective for time dependent PDEs as it forces propagation from the initial condition via causality

As a comparison you can turn the causal training on and off with the do_causal flag at the start of the file to see the difference in results.
We also show off the periodic boundary conditions here
'''

do_causal = True
print(f'Allen Cahn Causal Training Set to : {do_causal}')
torch.manual_seed(1234)

(xmin,xmax),(tmin,tmax) = (-1,1),(0,1)
domain = Domain2D(base = Rectangle(((xmin,tmin),(xmax,tmax))))

boundary_points = domain.generate_boundary_points(1000)
sampled_points = domain.generate_points(100_000)

x0 = boundary_points['exterior_edge_0']
x1 = boundary_points['exterior_edge_2']
t0= boundary_points['exterior_edge_3']

# Dataset and Loader
dataset = PINN_dataset()

dataset.add_group('boundary_0',x0,batch_size=100,shuffle = True)
dataset.add_group('boundary_1',x1,batch_size=100,shuffle = True)
dataset.add_group('t0',t0,batch_size=100,shuffle = True)
dataset.add_group('collocation points',sampled_points,batch_size=2000,shuffle=True,causal = do_causal)

# Equations And Losses
DL = PINN_Dataloader(dataset)

def u_IC(x):
    return x[:,0]**2*torch.cos(torch.pi*x[:,0])

def AllenCahn(u_t,u_xx,u,**kwargs):
    return u_t - 0.0001*u_xx + 5*u**3 -5*u

input_vars,output_vars = (['x','t'],['u'])
derivatives = get_derivatives(input_vars,output_vars,AllenCahn)

losses = Loss_handler(dataset)
losses.add_periodic('boundary_0','boundary_1','u')
losses.add_periodic('boundary_0','boundary_1','u_x')
losses.add_initial_condition('t0',{'u':u_IC})
losses.add_residual('collocation points',{'AllenCahn':AllenCahn})

#Network
net = Fourier_Net(2,1,100,4,RWF=True)
optimizer = torch.optim.Adam(params = net.parameters(), lr = 1e-3)
LR_sch = StepLR(optimizer,2000,0.9)

PINN = DE_Getter(net,input_vars,output_vars,derivatives)
net = net.cuda()

print(len(DL))
for epoch in range(0,1001):
    for x in DL:
        x = x.to('cuda')
        out = PINN(x)
        loss = losses.calculate(x,out,causal=do_causal,eps = 1)
        l = loss.individual_loss()
        ls = [l['initial condition'],l['periodic'],l['residual']]
        
        loss_sum = sum(ls)
        loss_sum.backward()
        optimizer.step()
        optimizer.zero_grad()
        LR_sch.step()
    
    loss.print_losses(epoch)


X,T = torch.meshgrid([torch.linspace(-1,1,100),torch.linspace(0,1,100)],indexing='ij')
xt = torch.stack([X.flatten(),T.flatten()],dim = -1)

with torch.no_grad():
    out = net.cpu()(xt)

plt.tricontourf(xt[:,1],xt[:,0],out[:,0],levels =100,cmap='jet')
plt.show()
