from torch_DE.equations import get_NavierStokes,get_derivatives
from torch_DE.geometry.shapes import *
from torch_DE.continuous import DE_Getter
from torch_DE.continuous.Networks import MLP,Wang_Net,Fourier_Net
from torch.optim.lr_scheduler import StepLR
from torch_DE.utils.data import PINN_Dataloader,PINN_dataset
from torch_DE.utils import GradNorm,Loss_handler,add_time
from torch_DE.post import Plotter,Evaluation,Tracker
import os
'''
Unsteady Vortex Shedding PINN Problem

This Problem solves unsteady flow across a cylinder at Re = 100. This example and initial conditions is from Wang et al from the repo: https://github.com/PredictiveIntelligenceLab/jaxpi

The flow is simulated across a T = 10. We break down the time into 10 [0,1] intervals and train a single network on each.

This problem demonstrates the following:
- Non-dimensionisation of the problem
    - significantly improves convergence rate

- Using multiple networks to time step forward
    - It is difficult to train the whole time interval. For each new interval, we set the initial contiion to be the output of the previous network at t=1
    - We also use the first network as the initial state for the other networks. This significantly improves convergence and is akin to finetuning later time intervals

- Using Finite difference rather than autograd for significantly faster training
    - We can get a significant increase in speed up for a modest decrease in accuracy. We could set up a loop to first train using finite difference and then further tune with autograd

- Using Data Driven constraint to represent the initial conditions
    - PINNs are overly dissapative so need suitable Initial conditions to create vortex shedding. Otherwise the PINN will collapse into some steady state solution

- Plotter Functionality in Torch_DE
    - Helps reduce the code needed to create graphs
'''
if not os.path.exists('Networks'):
    os.mkdir('Networks')

if not os.path.exists('Images'):
    os.mkdir('Images')

MAX_EPOCHS = 100

#Boolean if we need to non-dimensionalise
non_dim = True
L_star = 1
U_star = 1
if non_dim:
    L_star = 0.1

t_int = [0,1]
(xmin,xmax),(ymin,ymax) = (0,2.2/L_star), (0,0.41/L_star)
circle_center = (0.2/L_star,0.2/L_star)
circle_radius = 0.05/L_star

domain = Rectangle(((xmin,ymin),(xmax,ymax) ),'corners')
domain = Domain2D(base = domain)

hole = Circle(circle_center,r = circle_radius,num_points= 512)
domain.remove(hole,names= ['Cylinder'])

sampled_points = domain.generate_points(5_000_00)

domain.add_boundary_group('Cylinder','curve','Cyl_No_Slip')

num_points = 10_000
boundary_points = domain.generate_boundary_points(num_points=num_points,random = False)

inlet = boundary_points['exterior_edge_0']
outlet = boundary_points['exterior_edge_2']

top_wall = boundary_points['exterior_edge_1']
bot_wall = boundary_points['exterior_edge_3']
cyl = boundary_points['Cyl_No_Slip'].repeat(20,1)
#Merge the no slip boundaries together
no_slip = torch.cat([top_wall,bot_wall,cyl])

#Initial Conditon
def get_dataset():
    data = np.load("ns_unsteady.npy", allow_pickle=True).item()
    u_ref = np.array(data["u"])
    v_ref = np.array(data["v"])
    p_ref = np.array(data["p"])
    t = np.array(data["t"])
    coords = np.array(data["coords"])
    inflow_coords = np.array(data["inflow_coords"])
    outflow_coords = np.array(data["outflow_coords"])
    wall_coords = np.array(data["wall_coords"])
    cylinder_coords = np.array(data["cylinder_coords"])
    nu = np.array(data["nu"])
    # print(t)
    u_ref,v_ref,p_ref,coords,inflow_coords,outflow_coords,wall_coords,cylinder_coords,nu = (torch.tensor(t,dtype = torch.float32) for t in (u_ref,v_ref,p_ref,coords,inflow_coords,outflow_coords,wall_coords,cylinder_coords,nu))
    
    u0 = u_ref[-1,:]
    v0 = v_ref[-1,:]
    xy = coords
    return u0,v0,xy
u0,v0,x_IC = get_dataset()
#Rescale values if non-dimensionalising
u0,v0,x_IC = u0/U_star,v0/U_star,x_IC/L_star
x_IC = add_time('single point',x_IC,point = 0.0)

# Dataset and Loader
dataset = PINN_dataset()
dataset.add_group('inlet',inlet,batch_size=1000,shuffle = True)
dataset.add_group('no slip',no_slip,batch_size=1000,shuffle = True)
dataset.add_group('outlet',outlet,batch_size=1000,shuffle = True)
dataset.add_group('collocation points',sampled_points,batch_size=5000,shuffle=True)
dataset.add_time('random interval',[0,1])
# We add IC after setting the time for the other groups
dataset.add_group('initial condition',x_IC,{'u':u0,'v':v0},batch_size=1000,shuffle= True)

DL = PINN_Dataloader(dataset)

#Losses and Equations
Re = 100
input_vars,output_vars,derivatives,equations = get_NavierStokes(dims = 2,steady_state= False,Re=Re)
(NS_x,NS_y,incomp) = list(equations.values())

U = 1.5
u_inlet_func = lambda x : 4*U*x[:,1]*(ymax-x[:,1])/(ymax**2)
outlet_func = lambda x_dict,out_dict: 1/Re*out_dict['outlet']['u_x'] - out_dict['outlet']['p']

#Losses
losses = Loss_handler(dataset.groups)
losses.add_boundary('inlet',{'u':u_inlet_func,
                            'v':0})

losses.add_boundary('outlet',{'v_x':0})
losses.add_custom_function('outlet',{'outlet':(outlet_func,)})

losses.add_boundary('no slip',{'u':0,
                                'v':0 })

losses.add_residual('collocation points',{'stokes_x':NS_x,
                                        'stokes_y':NS_y, 
                                        'incomp':incomp },weighting=1)

losses.add_data_constraint('initial condition',['u','v'])

#Network, Optimizer and LR SETUP

#Post
plotter = Plotter(input_vars,output_vars)
plotter.contour_points_from_domain(domain,time=True)

for tp in range(0,1):
#PYTORCH SETUP
    torch.manual_seed(1234)
    net = Fourier_Net(3,3,128,4,RWF= True,activation= 'tanh')

    if tp > 0:
        net.load_state_dict(torch.load(f'Networks/Network_{tp-1}.pth'))
        # Set New IC
        with torch.no_grad():
            out = net.cpu()(x_IC.cpu())
            u0,v0 = out[:,0],out[:,1]
            dataset.add_group('initial condition',x_IC,{'u':u0,'v':v0},batch_size=1000,shuffle= True)

    optimizer = torch.optim.Adam(params = net.parameters(), lr = 1e-3)
    LR_sch = StepLR(optimizer,10000,0.9)
    net = net.cuda()

    PINN = DE_Getter(net)
    PINN.set_vars(input_vars,output_vars)
    PINN.set_derivatives(derivatives)
    PINN.set_deriv_method('FD')

    # Training Loop
    weights = torch.ones(len(losses),dtype = torch.float32)
    print(f'Num Batches {len(DL)}')
    for epoch in range(1,MAX_EPOCHS+1):
        for x in DL:
            x = x.to(device = 'cuda')
            #Calculate Derivatives
            out = PINN.calculate(x)
            #Get Losses
            loss = losses(x,out)
            loss_sum = sum([w*l for w,l in zip(weights,loss.MSE(flatten= True))])

            if (epoch % 5) == 0 and (epoch > 0):
                weights = GradNorm(net,weights,*loss.MSE(flatten= True))
                
            loss_sum.backward()

            optimizer.step()
            LR_sch.step()

            optimizer.zero_grad()

        loss.print_losses(epoch)
        if (epoch % MAX_EPOCHS) == 0:
            for time_point in [0.0,1]:
                with torch.no_grad():
                    out = net.cpu()(x_IC)
                    t_str = str(time_point).replace('.','_')
                    plotter.set_time_point(time_point)
                    plotter.contour(net,['x','y'],'u',f'u velocity at time {t_str} for period {tp}')
                    # plotter.show()
                    plotter.savefig(f'Images/Contour_u_time_{t_str}_period_{tp}.png')
                net = net.cuda()
            
    #End of training Loop:
    torch.save(net.state_dict(),f'Networks/Network_{tp}.pth')