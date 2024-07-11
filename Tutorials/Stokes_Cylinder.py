from torch_DE.equations import get_NavierStokes,get_derivatives
from torch_DE.geometry.shapes import *
from torch_DE.continuous import DE_Getter
from torch_DE.continuous.Networks import MLP,Wang_Net
from torch.optim.lr_scheduler import StepLR
from torch_DE.utils.data import PINN_Dataloader,PINN_dataset
from torch_DE.utils import GradNorm,Loss_handler

'''

'''

(xmin,xmax),(ymin,ymax) = (0,1), (0,0.41)
domain = Rectangle(((xmin,ymin),(xmax,ymax) ),'corners')
domain = Domain2D(base = domain)

hole = Circle((0.2,0.2),r = 0.05,num_points= 512)
domain.remove(hole,names= ['Cylinder'])

sampled_points = domain.generate_points(400_000)

domain.add_boundary_group('Cylinder','curve','Cyl_No_Slip')

num_points = 1000
boundary_points = domain.generate_boundary_points(num_points=num_points,random = False)

inlet = boundary_points['exterior_edge_0']
outlet = boundary_points['exterior_edge_2']

top_wall = boundary_points['exterior_edge_1']
bot_wall = boundary_points['exterior_edge_3']
cyl = boundary_points['Cyl_No_Slip']
no_slip = torch.cat([top_wall,bot_wall,cyl])


# Dataset and Loader
dataset = PINN_dataset()
dataset.add_group('inlet',inlet,batch_size=100,shuffle = True)
dataset.add_group('no slip',no_slip,batch_size=100,shuffle = True)
dataset.add_group('outlet',outlet,batch_size=100,shuffle = True)
dataset.add_group('collocation points',sampled_points,batch_size=2000,shuffle=True)

DL = PINN_Dataloader(dataset)

#Losses and Equations

input_vars,output_vars,derivatives,equations = get_NavierStokes(dims = 2,steady_state= True,Re=100)
(NS_x,NS_y,incomp) = list(equations.values())

U = 0.3
u_inlet_func = lambda x : 4*U*x[:,1]*(0.41-x[:,1])/(0.41**2)

losses = Loss_handler(dataset.groups)
losses.add_boundary('inlet',{'u':u_inlet_func,
                            'v':0})

losses.add_boundary('outlet',{'u':u_inlet_func,
                            'v':0})


losses.add_boundary('no slip',{'u':0,
                                'v':0 })

losses.add_residual('collocation points',{'stokes_x':NS_x,
                                        'stokes_y':NS_y, 
                                        'incomp':incomp },weighting=1)


#PYTORCH SETUP
torch.manual_seed(1234)
net = MLP(2,3,128,6,activation='sin')
optimizer = torch.optim.Adam(params = net.parameters(), lr = 1e-3)
LR_sch = StepLR(optimizer,10000,0.9)
net = net.cuda()

PINN = DE_Getter(net)
PINN.set_vars(input_vars,output_vars)
PINN.set_derivatives(derivatives)

# Training Loop
weights = torch.ones(len(losses),dtype = torch.float32)
print(f'Num Batches {len(DL)}')
for epoch in range(1001):
    weight_flag = True
    for x in DL:
        x = x.to(device = 'cuda')
        #Calculate Derivatives
        out = PINN.calculate(x)
        #Get Losses
        loss = losses(x,out)
        loss_sum = sum([w*l for w,l in zip(weights,loss.MSE(flatten= True))])

        if (epoch % 5) == 0 and weight_flag:
            weights = GradNorm(net,weights,*loss.MSE(flatten= True))
            weight_flag = False
        loss_sum.backward()

        optimizer.step()
        LR_sch.step()

        optimizer.zero_grad()

    l = loss.individual_loss()
    print(f"Epoch {epoch} Loss {float(loss_sum):.3E} Boundary {float(l['boundary']):.3E}, Residual {float(l['residual']):.3E} ")

