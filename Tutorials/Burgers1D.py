from torch_DE.geometry.shapes import *
from torch_DE.continuous import DE_Getter
from torch_DE.continuous.Networks import MLP
from torch.optim.lr_scheduler import StepLR
from torch_DE.utils.data import PINN_Dataloader,PINN_dataset
from torch_DE.utils import Loss_handler
from matplotlib import tri
'''
Burgers Equation for PINN

We solve the following PDE:
    u_t + u*(u_x) - nu*u_xx = 0

    for x in [-1,1], t in [0,1]

with the following conditions:
    nu = 0.01/pi.
    u(x,0) = -sin(pi*x)
    u(-1,t) = u(1,t) = 0

In the losses, we weight the initial and boundary condition by 100 for better convergence
'''


#GGenerate points from Geometry
xmin,xmax = (-1,1)
tmin,tmax = (0,1)

domain = Rectangle((( xmin,tmin),(xmax,tmax) ),'corners')
domain = Domain2D(base = domain)
sampled_points = domain.generate_points(50_000)

num_points = 100

x0 = domain.generate_points_from_boundary('exterior_edge_0',num_points)
x1 = domain.generate_points_from_boundary('exterior_edge_2',num_points)
t0= domain.generate_points_from_boundary('exterior_edge_3',num_points)

# Dataset and Loader
dataset = PINN_dataset()
dataset.add_group('initial condition',t0,batch_size=num_points)
dataset.add_group('x0',x0,batch_size=num_points)
dataset.add_group('x1',x1,batch_size=num_points)
dataset.add_group('collocation points',sampled_points,batch_size=2000,shuffle=True)
DL = PINN_Dataloader(dataset)

#Network
net = MLP(2,1,100,4)
optimizer = torch.optim.Adam(params = net.parameters(), lr = 1e-3)
LR_sch = StepLR(optimizer,10000,0.9)
net = net.cuda()

PINN = DE_Getter(net)
PINN.set_vars(['x','t'],['u'])
PINN.set_derivatives(['u_x','u_t','u_xx'])
PINN.set_deriv_method('AD')

#Losses
def burgers(u,u_t,u_x,u_xx,**kwargs):
    return u_t + u*u_x - (0.01/torch.pi)*u_xx

losses = Loss_handler(dataset)
losses.add_initial_condition('initial condition',{'u': lambda x : -torch.sin(torch.pi*x[:,0]) })
losses.add_boundary('x0',{'u':0})
losses.add_boundary('x1',{'u':0})
losses.add_residual('collocation points',{'burgers1D':burgers})

# Training Loop
print(f'Num Batches {len(DL)}')
for epoch in range(1001):
    for x in DL:
        x = x.to(device = 'cuda')
        #Calculate Derivatives
        out = PINN.calculate(x)
        #Get Losses
        loss = losses(x,out).individual_loss()
        loss = 100*loss['initial condition'] + loss['residual'] + 100*loss['boundary']
        loss.backward()

        optimizer.step()
        LR_sch.step()

        optimizer.zero_grad()
    if (epoch % 100) == 0:
        print(f'Epoch {epoch} Loss {float(loss):.3E} Learning Rate: {float(LR_sch.get_last_lr()[0]):.3E}')


with torch.no_grad():
    xy = torch.meshgrid(torch.linspace(xmin,xmax,100),torch.linspace(tmin,tmax,100))
    x = xy[0].flatten()
    y = xy[1].flatten()

    tri_ang = tri.Triangulation(y, x)

    net = net.cpu()
    u = net(torch.stack([x,y],-1))

    plt.title("Burger's Equation")
    plt.tricontourf(tri_ang,u[:,0],levels =100,cmap = 'jet')
    plt.colorbar()
    plt.xlabel('time')
    plt.ylabel('x')
    plt.show()