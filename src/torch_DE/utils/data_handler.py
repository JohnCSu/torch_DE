
import torch
from torch_DE.utils.sampling import add_random_time,add_random_time_point
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


    def set_to_device(self,device,to_show = False):
        if to_show:
            print(f'Set all tensors to device {device}')
        for group in self.keys():
            self[group] = self[group].to(device) 

        self.device = device

    def add_time(self,*,dim = -1,ignore_groups = None):
        #Ignored groups do not get resampled when time_resample is called!
        self.time_states ={}

        if isinstance(ignore_groups,str):
            ignore_groups = [ignore_groups]

        for key in self.keys():
            if key not in ignore_groups:
                tensor = self[key]
                self[key] =torch.cat([tensor,torch.zeros((tensor.shape[0],1),device=tensor.device)],dim = dim)
                self.time_states[key] = 'all'

    def set_time_sampling(self,*keys,sampling_method = 'all'):
        '''
        Set the sampling method for different groups.

        sampling method:
            - 'all': all time points are sampled randomly along the interval
            - 'point': A single random time point from the interval is assigned to the group
        '''
        if (sampling_method != 'all') and (sampling_method != 'point'):
            raise ValueError(f'sampling_method takes the strings of "all" or "point"')
        
        for key in keys:
            self.time_states[key] = sampling_method

    def time_resample(self,interval):
        for key in self.time_states.keys():
            if self.time_states[key] == 'all':
                add_random_time(self[key],interval)
            else:
                add_random_time_point(self[key],interval)