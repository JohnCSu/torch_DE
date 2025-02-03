from torch_DE.equations import get_NavierStokes,get_derivatives,DE_func
from torch_DE.geometry.shapes import *
from torch_DE.continuous import DE_Getter
from torch_DE.continuous.Networks import Fourier_Net
from torch.optim.lr_scheduler import StepLR
from torch_DE.utils.data import PINN_Dataloader,PINN_dataset
from torch_DE.utils import GradNorm,Loss_handler


non_dim = True
L_star = 1
U_star = 1
Re = 1/0.001

if non_dim:
    L_star = 0.1
    U_star = 0.2
    Re = 20
t_int = [0,1]

(xmin,xmax),(ymin,ymax) = (0,2.2/L_star), (0,0.41/L_star)
circle_center = (0.2/L_star,0.2/L_star)
circle_radius = 0.05/L_star

domain = Rectangle(((xmin,ymin),(xmax,ymax) ),'corners')
domain = Domain2D(base = domain)

hole = Circle(circle_center,r = circle_radius,num_points= 512)
domain.remove(hole,names= ['Cylinder'])

sampled_points = domain.generate_points(400_000)

domain.add_boundary_group('Cylinder','curve','Cyl_No_Slip')
sdf = DE_func(domain.create_sdf(device='cuda'))
num_points = 1000
boundary_points = domain.generate_boundary_points(num_points=num_points,random = False)

inlet = boundary_points['exterior_edge_0']
outlet = boundary_points['exterior_edge_2']

top_wall = boundary_points['exterior_edge_1']
bot_wall = boundary_points['exterior_edge_3']
cyl = boundary_points['Cyl_No_Slip']
no_slip = torch.cat([top_wall,bot_wall,cyl])


#Equations

input_vars,output_vars,derivatives,equations = get_NavierStokes(dims = 2,steady_state= True,Re=Re)
(NS_x,NS_y,incomp) = list(equations.values())


# Dataset and Loader
dataset = PINN_dataset(input_vars)
dataset.add_group('inlet',inlet,batch_size=200,shuffle = True)
dataset.add_group('no slip',no_slip,batch_size=200,shuffle = True)
dataset.add_group('outlet',outlet,batch_size=200,shuffle = True)
dataset.add_group('collocation points',sampled_points,batch_size=2_000,shuffle=True)

DL = PINN_Dataloader(dataset)

U = 0.3/U_star

u_inlet_func = DE_func(lambda y,**kwargs : 4*U*y*(ymax-y)/(ymax**2))
losses = Loss_handler(dataset)

losses.add_boundary('inlet',{'u':u_inlet_func,
                            'v':0})

losses.add_boundary('outlet',{'u':u_inlet_func,
    'v':0})


losses.add_boundary('no slip',{'u':0,
                                'v':0 })

losses.add_residual('collocation points',{'stokes_x':NS_x,
                                        'stokes_y':NS_y, 
                                        'incomp':incomp },weighting=sdf)


#PYTORCH SETUP
torch.manual_seed(1234)
net = Fourier_Net(2,3,128,6,activation='sin',RWF=True)
optimizer = torch.optim.Adam(params = net.parameters(), lr = 1e-3)
LR_sch = StepLR(optimizer,10000,0.9)
net = net.cuda()

PINN = DE_Getter(net)
PINN.set_vars(input_vars,output_vars)
PINN.set_derivatives(derivatives)

net = net.cuda()
# Training Loop
weights = torch.ones(len(losses),dtype = torch.float32,device = 'cuda')
print(f'Num Batches {len(DL)}')
for epoch in range(0,101):
    weight_flag = True
    for x in DL:
        x = x.to(device = 'cuda')
        #Calculate Derivatives
        out = PINN.calculate(x)
        #Get Losses
        loss = losses(x,out)
        loss_sum = sum(loss.individual_losses() * weights)

        if (epoch % 10) == 0 and epoch > 0 and weight_flag is True:
            weights = GradNorm(net,weights,*loss.individual_losses())
            weight_flag = False
            print(weights)

        loss_sum.backward()

        optimizer.step()
        LR_sch.step()

        optimizer.zero_grad()

    loss.print_losses(epoch)



with torch.no_grad():
    plot_points = domain.generate_points(10_000)
    #B,x,y points
    x,y = plot_points[:,0],plot_points[:,1]
    out= net(plot_points.cuda()).cpu()
    # plt.scatter(x,y,s=2)
    plt.tricontourf(x,y,out[:,0],levels =100,cmap = 'jet')
    plt.gca().set_aspect('equal')
    plt.show()