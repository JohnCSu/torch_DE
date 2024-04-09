import torch
from matplotlib import pyplot as plt

def sample_from_tensor(num_points: int,t:torch.Tensor,dim:int = 0):
    '''
    randomly samples `num_points` from the tensor `t`. Always indexes from the first (batch) dimension so samples a `(num_points,D,...)` from tensor `t` of size `(L,D,...)`
    '''
    return t[torch.randint(low =0,high = t.shape[dim],size = (num_points,))]


class R3_sampler():
    def __init__(self,sampler,device = 'cpu') -> None:
        '''
        Sampler Based on the Retain, Resample and Release Algorithim by __ et al

        This is modified to use L2 loss as it means we don't have to recompute means and all that - a small increase in speed and cleanliness

        inputs:
        sampler: A function that takes in the arguement of the num of points to generate and other keywords i.e. sampler(num_points,**kwargs)
        device: The device to put the new sample of collocation point to. Default is cuda
        
        '''
        
        self.sampler = sampler
        self.device = device
    @staticmethod
    def F_measure(*res,device = 'cpu'):
        return torch.sum(torch.stack([torch.abs(r) for r in res],dim = 0),dim=0).to(device)

    def RRR_sample(self,x:torch.tensor,*res,plot_epoch = None,**kwargs):
        
        with torch.no_grad():
            F_measure = self.F_measure(*res,device = self.device)
            x = x.to(self.device)
            # print(F_measure.shape)
            mean = F_measure.mean()
            #Retain
            x_retain = self.retain(x,F_measure,mean)
            
            #Resample
            num_new_points = x.shape[0] - x_retain.shape[0]
            x_new = self.resample(num_new_points,**kwargs)

            #Plotting (Optional):
            if plot_epoch is not None:
                self.plot(plot_epoch,x_retain,x_new,F_measure)
            
            # Release (Returns the resampled collocation points and )
            return torch.cat([x_retain.to(self.device),x_new.to(self.device)],dim = 0)

    def retain(self,x,Res,mean):
        #We only want the points that are greater than the mean
        return x[Res >= mean]

    def resample(self,num_points,**kwargs):
        return self.sampler(num_points,**kwargs)

    def __call__(self,x:torch.tensor,*res,plot_epoch = None,**kwargs) -> torch.tensor:
        return self.RRR_sample(x,*res,plot_epoch = plot_epoch,**kwargs)
    
    def plot(self,epoch,retained_points,new_points,F_measure):
            plt.scatter(retained_points[:,0],retained_points[:,1],s = 3,c='b',label = f'Retained Points N{torch.sum(F_measure >= F_measure.mean())}' )
            plt.scatter(new_points[:,0],new_points[:,1],s = 3,c = 'r',label = f'{torch.sum(F_measure < F_measure.mean())} New Points After Resampling ')
            plt.title(f'Sample vs ReSample at Iteration {epoch} For F Mean Criteria: {F_measure.mean():.3E}')
            
            plt.legend(loc='best', bbox_to_anchor=(0.33, -0.1))
            plt.show()


class Data_handler(dict):
    def __init__(self):
        super().__init__()
        self.device = 'cpu'
    def merge_groups(self):
        '''
        Create a concatenated vector in the group 'all'. Also creates the equivalent group names, and sizes to pass into DE_Getter()
        '''
        self.pop('all',None)
        self.set_to_device(self.device,to_show=False)
        group_names,group_sizes,group_data = zip(*[(name,data.shape[0],data) for name,data in self.items()])
       
        self['all'] = torch.cat(group_data)
        return self['all'],group_names,group_sizes

    def group_names(self):
        return list(self.keys())


    def set_to_device(self,device,to_show = True):
        if to_show:
            print(f'Set all tensors to device {device}')
        for group in self.keys():
            self[group] = self[group].to(device) 

        self.device = device
