from torch_DE.geometry.shapes import *
from torch_DE.continuous import DE_Getter
from torch_DE.continuous.Networks import MLP
from torch_DE.continuous.utils import *
from torch_DE.continuous.Engines import FD_engine
from torch_DE.Equations import get_NavierStokes
from torch.optim.lr_scheduler import StepLR
from matplotlib.patches import Circle as pltCircle
#Geometry
(xmin,xmax),(ymin,ymax) = (0,1), (0,0.41)
domain = Rectangle(((xmin,ymin),(xmax,ymax) ),'corners')
domain = Domain2D(base = domain)
hole = Circle((0.2,0.2),r = 0.05,num_points= 512)
domain.remove(hole,names= ['Cylinder'])
sdf = domain.create_sdf(resolution = 100)

#Create Data Points
sampled_points = domain.generate_points(400_000)
domain.add_boundary_group('Cylinder','Cyl_No_Slip')
b_groups =domain.boundary_groups 
cyl = b_groups['Cyl_No_Slip']
cyl_points = torch.tensor(list(cyl.coords))
inlet = domain.generate_points_from_line(b_groups['exterior_edge_0'],100,random = False)
outlet = domain.generate_points_from_line(b_groups['exterior_edge_2'],100,random = False)
top = domain.generate_points_from_line(b_groups['exterior_edge_1'],100,random = False)
bot = domain.generate_points_from_line(b_groups['exterior_edge_3'],100,random = False)

#Equations
input_vars,output_vars,derivatives,equations = get_NavierStokes(dims = 2,steady_state= True,Re=100)
(NS_x,NS_y,incomp) = list(equations.values())
input_vars,output_vars,sorted(derivatives)

#Inlet Condition
U = 0.3
u_inlet_func = lambda x : 4*U*x[:,1]*(0.41-x[:,1])/(0.41**2)

#DATA
data  = Data_handler()
data['collocation points'] = sample_from_tensor(2000,sampled_points,0)
data['inlet'] = inlet
data['outlet'] = outlet
data['no_slip_walls'] = torch.cat([top,bot],dim = 0)
data['cylinder'] = cyl_points
data.set_to_device('cuda')
#PYTORCH SETUP
torch.manual_seed(1234)
net = MLP(2,3,128,6,activation='sin')
optimizer = torch.optim.Adam(params = net.parameters(), lr = 1e-3)
LR_sch = StepLR(optimizer,10000,0.9)

#SET UP DERIVATIVES
PINN = DE_Getter(net)
PINN.set_vars(input_vars,output_vars)
PINN.set_derivatives(derivatives)
FD = FD_engine(net = net,derivatives= PINN.derivatives,dxs = [0.01,0.01],sdf= sdf)
PINN.set_deriv_method(FD)

#LOSSES
losses = Loss_handler(data.group_names())
losses.add_boundary('inlet',{'u':u_inlet_func,
                             'v':0})
losses.add_boundary('outlet',{'u':u_inlet_func,
                             'v':0})
losses.add_boundary('cylinder',{'u':0,
                                'v':0 })
losses.add_boundary('no_slip_walls',{'u':0,
                                     'v':0 })
losses.add_residual('collocation points',{'stokes_x':NS_x,
                                          'stokes_y':NS_y, 
                                          'incomp':incomp },weighting=sdf)
#SAMPLING STRATEGY
sampler = R3_sampler(sample_from_tensor)

net= net.cuda()
for i in range(100_000):
    data.set_to_device('cuda',to_show= False)
    output = PINN.calculate(data,target_group = 'collocation points')
    res = sum([losses.loss_groups['Residual']['collocation points'][i](data,output) for i in ['stokes_x','stokes_y','incomp']]) 

    loss = losses.calculate(data,output)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    LR_sch.step()

    data['collocation points'] = sampler(data['collocation points'],res,t=sampled_points)
    if (i % 500) == 0:
        losses.print_losses(i)

torch.save(net.state_dict(),'Cylinder_PINN.pth')