from torch_DE.geometry.shapes import *
from torch_DE.continuous import DE_Getter
from torch_DE.continuous.Networks import MLP
from torch.optim.lr_scheduler import StepLR
from torch_DE.utils.data import PINN_Dataloader,PINN_dataset
from torch_DE.utils import Loss_handler
from torch_DE.geometry.shapes import Domain2D,Rectangle
from torch_DE.equations import get_derivatives,DE_func
'''
Lid Driven Cavity Example. This follows from the example given by Nvidia Modulus found here: 
https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/basics/lid_driven_cavity_flow.html

This Example demonstrates the spatial weighting functionality of torch_DE and the signed distance function or sdf generated function from the geometry module to be used for spatial weighting of points
'''

xmin,xmax = (-0.1,0.1)
ymin,ymax = (-0.1,0.1) 
domain = Domain2D(base = Rectangle(((xmin,ymin),(xmax,ymax) ),'corners'))
sdf =DE_func(domain.create_sdf(device = 'cuda'))

sampled_points = domain.generate_points(100_000)

num_points = 10000
left_wall = domain.generate_points_from_boundary('exterior_edge_0',num_points)
top_wall = domain.generate_points_from_boundary('exterior_edge_1',num_points)
right_wall = domain.generate_points_from_boundary('exterior_edge_2',num_points)
bot_wall = domain.generate_points_from_boundary('exterior_edge_3',num_points)

no_slip = torch.cat([left_wall,right_wall,bot_wall],dim = 0)

#Losses, And Equations
@DE_func
def Stokes_Flow_x(u,v,u_x,u_y,u_xx,u_yy,p_x,v_y,v_xx,v_yy,p_y,Re = 100, **kwargs):
    NS_x =u*u_x + v*u_y + p_x - 1/Re*(u_xx + u_yy)
    return NS_x

@DE_func
def Stokes_Flow_y(u,v,u_x,u_y,u_xx,u_yy,p_x,v_y,v_x,v_xx,v_yy,p_y,Re = 100, **kwargs):
    NS_y =u*v_x + v*v_y + p_y - 1/Re*(v_xx + v_yy)
    return NS_y
@DE_func
def incomp(u_x,v_y, **kwargs):
    incomp = u_x + v_y
    return incomp

input_vars = ['x','y']
output_vars = ['u','v','p']
derivative_names = get_derivatives(input_vars,output_vars,Stokes_Flow_x,Stokes_Flow_y,incomp,merge = True)

# Dataset and Loader
dataset = PINN_dataset(input_vars)
dataset.add_group('inlet',top_wall,batch_size=1000,shuffle = True)
dataset.add_group('no slip',no_slip,batch_size=1000,shuffle = True)
dataset.add_group('collocation points',sampled_points,batch_size=2000,shuffle=True)
DL = PINN_Dataloader(dataset)

inlet_weight_func = DE_func(lambda x,**kwargs: 1-10*torch.abs(x))

losses = Loss_handler(dataset)
losses.add_boundary('no slip',{'u':0,
                               'v':0})
losses.add_boundary('inlet',{'u':1,
                               'v':0},weighting = {'u':inlet_weight_func,'v':1})

losses.add_residual('collocation points',{'stokes_x':Stokes_Flow_x,
                                          'stokes_y':Stokes_Flow_y, 
                                          'incomp':incomp },weighting = sdf)

#Network
net = MLP(2,3,100,6)
optimizer = torch.optim.Adam(params = net.parameters(), lr = 1e-3)
LR_sch = StepLR(optimizer,10000,0.9)
net = net.cuda()

PINN = DE_Getter(net)
PINN.set_vars(['x','y'],['u','v','p'])
PINN.set_derivatives(list(derivative_names))
PINN.set_deriv_method('AD')


# Training Loop
print(f'Num Batches {len(DL)}')
net = net.cuda()

for epoch in range(101):
    for x in DL:
        x = x.to(device = 'cuda')
        #Calculate Derivatives
        out = PINN.calculate(x)
        #Get Losses
        loss = losses(x,out)
        type_losses = loss.grouped_losses('loss_type')

        loss_sum =  type_losses['residual'] + 100*type_losses['boundary']
        loss_sum.backward()

        optimizer.step()
        LR_sch.step()

        optimizer.zero_grad()
    # if (epoch % 100) == 0:
    loss.print_losses(epoch)


with torch.no_grad():
    xy = torch.meshgrid(torch.linspace(xmin,xmax,100),torch.linspace(ymin,ymax,100))
    x = xy[0].flatten()
    y = xy[1].flatten()

    net = net.cpu()
    u = net(torch.stack([x,y],-1))

    plt.title("v for LDC Re=100")
    plt.tricontourf(x,y,u[:,0],levels =100,cmap = 'jet')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()