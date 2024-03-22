
import torch
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

