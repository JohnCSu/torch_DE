import torch
from matplotlib import pyplot as plt
from typing import Callable,Union,Dict,Iterable
from torch_DE.utils.loss import Loss
import torch
from matplotlib import pyplot as plt
from typing import Callable,Union,Dict,Iterable



def add_time_col(tensor,col = -1):
    return torch.cat([tensor,torch.zeros((tensor.shape[0],1),device=tensor.device)],dim = col)

def add_time_point(tensor,t_point,axis = -1):
    tensor[:,axis] = t_point*torch.ones((tensor.shape[0]),device=tensor.device)


def add_random_time_point(tensor,interval,col = -1):
    a,b = interval 
    tensor[:,col] = torch.ones((tensor.shape[0]),device=tensor.device)*(torch.rand(1,device=tensor.device)*(b-a)+a)

def add_random_time(tensor,interval,col = -1):
    a,b = interval
    tensor[:,col] = torch.rand((tensor.shape[0]),device=tensor.device)*((b-a)+ a)


def sample_from_tensor(num_points: int,t:torch.Tensor,causal = False,time_col:int = -1):
    '''
    randomly samples `num_points` from the tensor `t`. Always indexes from the first (batch) dimension so samples a `(num_points,D,...)` from tensor `t` of size `(L,D,...)`
    '''
    tensor =t[torch.randint(low =0,high = t.shape[0],size = (num_points,))]
    return tensor if not causal else tensor[tensor[:,time_col].sort()[-1]]


class R3_sampler():
    def __init__(self,sampler:Callable,*,group:dict = None,device:str = 'cpu',causal = False,time_interval = None) -> None:
        '''
        Sampler Based on the Retain, Resample and Release Algorithim by __ et al

        inputs:
        group: group to perform the R3 Sampling on
        sampler: A function that takes in the arguement of the num of points to generate and other keywords i.e. sampler(num_points,**kwargs)
        device: The device to put the new sample of collocation point to. Default is cuda
        funcs: The resiudal functions to call
        
        causal flag. Set true if causality weighting is to be used. This is independent of the causal gate. See time_interval for causal gate
        time_interval: Tuple representing the start and end points of the time interval. Set for activating causal gate mechnaism
        '''
        self.group = group
        self.sampler = sampler
        self.device = device
        self._plot_args = None

        self.time_interval = time_interval
        self.g = lambda t,g: t
        self.alpha =5
        self.gamma = -0.5
        self.nu = 1e-3
        self.eps = 20
        self.dmax = 0.1
        self.causal = causal
        if self.time_interval is not None:
            T = self.time_interval[-1]
            self.g = lambda t,g : (1-torch.tanh(self.alpha*(t/T-g)))/2
    def __call__(self,x:torch.tensor,res:Union[Iterable,Loss],loss_type = 'weighted',**kwargs) -> torch.tensor:
        '''
        Generate New points based on R3 Sampling

        inputs:
            x: torch.Tensor of data points used in current network evaluation shoud be of `shape (N,D)` with N being the batch dimension
            
            res: Iterable | loss object. If iterable then we have a list/tuple of residuals vectors to use as the F measure.
                if loss object then torch_DE automatically extracts the residuals from group. The L1 norm is applied first before summing all residual terms together.
                if x and res are not `dict` or `Loss()` objects then the group and loss_type defined in the sampler is ignored
            
            loss_type (string): Type of residual loss to use if Loss object is given, options available are 'weighted' or 'point error'
        returns:
            x_new: torch.Tensor of newly sampled points based on R3 algorithim. Same shape as x
        '''
        return self.RRR_sample(x,res,loss_type,**kwargs)
    
    @staticmethod
    def F_measure(*res,device = 'cpu'):
        return torch.sum(torch.stack([torch.abs(r) for r in res],dim = 0),dim=0).to(device)

    def RRR_sample(self,x:Union[dict,torch.Tensor],res:Union[list,Loss],loss_type = 'weighted',**kwargs):

        if isinstance(res,Loss):
            if loss_type == 'weighted':
                res = list(res.weighted_error()['Residual'][self.group].values())
            elif loss_type == 'point error':
                res = list(res.point_error()['Residual'][self.group].values())
            else:
                raise ValueError(f'loss_type accepts only strings weighted and point error')
            
        if isinstance(x,dict):
            x = x[self.group]

        with torch.no_grad():
            causal_gate = self.g(x[:,-1],self.gamma)
            F_measure = self.F_measure(*res,device = self.device)*causal_gate
            x = x.to(self.device)
            # print(F_measure.shape)
            mean = F_measure.mean()
            #Retain
            x_retain = self.retain(x,F_measure,mean)
            
            #Resample
            num_new_points = x.shape[0] - x_retain.shape[0]
            x_new = self.resample(num_new_points,**kwargs)

            #Plotting (Optional):
            self._plot_args = [x_retain,x_new,F_measure]
            #Causal Gate for Gamma
            if self.time_interval is not None: 
                L = mean
                self.gamma = self.gamma + self.nu*min(torch.exp(-self.eps*L),self.dmax)

            # Release (Returns the resampled collocation points and )
            Release = torch.cat([x_retain.to(self.device),x_new.to(self.device)],dim = 0)
            return Release if self.causal is False else Release[Release[:,-1].sort()[1]]

    def retain(self,x,Res,mean):
        #We only want the points that are greater than the mean
        return x[Res >= mean]

    def resample(self,num_points,**kwargs):
        return self.sampler(num_points,**kwargs)



    def plot(self,epoch,show = True,save_name = None,aspect_ratio = 'auto',transpose_axis = False,**kwargs:dict):
            
            plt.clf()
            plt.cla()
            kwargs.setdefault('s',3)
            retained_points,new_points,F_measure = self._plot_args

            if transpose_axis:
                yr,xr = retained_points[:,0].cpu(),retained_points[:,1].cpu()
                yn,xn = new_points[:,0].cpu(),new_points[:,1].cpu()
            else:
                xr,yr = retained_points[:,0].cpu(),retained_points[:,1].cpu()
                xn,yn = new_points[:,0].cpu(),new_points[:,1].cpu()
                
            plt.scatter(xr,yr,c= 'b', label = f'Retained Points N{torch.sum(F_measure >= F_measure.mean())}',**kwargs )
            plt.scatter(xn,yn,c = 'r',label = f'{torch.sum(F_measure < F_measure.mean())} New Points After Resampling',**kwargs)
            plt.title(f'Sample vs ReSample at Iteration {epoch} For F Mean Criteria: {F_measure.mean():.3E}')
            plt.gca().set_aspect(aspect_ratio)
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25))
            
            if save_name is not None:
                plt.savefig(save_name,bbox_inches = 'tight')
            if show:
                plt.show()
            plt.clf()
            plt.cla()
