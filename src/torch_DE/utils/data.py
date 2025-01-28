
import torch
from torch.utils.data import Dataset,DataLoader,Sampler
import numpy as np
import warnings
from typing import Dict,List,Tuple,Union,Iterator
import inspect
from torch_DE.utils.time import add_time
from torch_DE.symbols import Variable_dict
from torch import Tensor

class PINN_group():
    def __init__(self,name:str,inputs:Tensor,batch_size:int,targets:Union[torch.Tensor,dict],input_vars:list[str],*,shuffle = False):
        '''
        Container for data for a defined grouped

        inputs:
            - name : str name of group
            - inputs: Tensor: full tensor of group of size NxD where N is the number of input points and D is the dimensional input of the network
            - batch_size: int batch size. must be smaller than N
            - targets: Tensor | dict: target data that the output of the network with respect to this specific group must match.
            - variables: list[str] | None: variable names of each input dimension. should be the same size as D 

        '''
        self.name:str = name
        self.batch_size:int = batch_size
        self.N,self.D = inputs.shape

        self.input_vars:list[str] = input_vars
        self.inputs:dict = {'input': inputs}
        self.inputs.update({input_var:inputs[:,i] for i,input_var in enumerate(input_vars)})
        self.has_multiple_inputs = False
        self.checks()

        if isinstance(targets,dict):
            self.targets = PINN_dict(targets)
        else:
            self.targets = PINN_dict({'target': targets}) if isinstance(targets,torch.Tensor) else None
        
        self.shuffle = shuffle 

        

    def __len__(self):
        return int(self.N)

    def checks(self):
        assert len(self.inputs['input'].shape) == 2
        assert self.batch_size <= self.__len__()
        assert len(self.input_vars) == self.D


    def to(self,*args,**kwargs):
        self.inputs = {key:x.to(*args,**kwargs) for key,x in self.inputs.items()}
        if self.targets is not None:
            self.targets = {key:x.to(*args,**kwargs) for key,x in self.targets.items()}
        return self
    

    def subgroup(self,idx):
        '''
        Given an list of indices, return a smaller PINN group containing data only on those indices. This is used for batching each group
        '''
        inputs = self.inputs['input'][idx]
        
        if self.has_multiple_inputs:
            inputs = tuple(x[idx] for x in self.inputs)
        
        targets = None
        if self.targets is not None:
            targets = {target: target_tensor[idx] for target,target_tensor in self.targets.items()}

        return PINN_group(self.name,inputs,self.batch_size,targets,input_vars=self.input_vars,shuffle=self.shuffle)


    def add_time(self,time_type,time_interval:Union[list,tuple] = None,point:float = None,dim:int = -1):
        if self.has_multiple_inputs:
            for tensor in self.inputs[1:]:
                assert tensor.shape == self.inputs[0].shape, 'tensors in each input need to be of same shape. You may need to manually assign time to each individual inputs'
            tensors = self.inputs
        else:
            tensors = [self.inputs]
        self.inputs = add_time(time_type,*tensors,time_interval =time_interval,point=point,dim=dim)
        

    # def __getitem__(self,idx):
    #     return self.subgroup(idx)
    def __getitem__(self,key):
        return self.inputs[key]


class PINN_dict(Variable_dict):
    '''
    Basically a regular dict but with some extra functionality e.g .to() to help cast tensors to devices as well as keys using `sympy.Symbol()` see `torch_DE.symbols.Variable_dict` for moore info
    '''
    def to(self,*args, **kwargs):
        '''
        Modify properties of each tensor in this dictionary. See Tensor.to() method in pytorch for available arguments
        '''
        for key in self.keys():
            self[key] = self[key].to(*args,**kwargs)
        return self

class PINN_dataset(Dataset):
    '''
    Dataset Class for PINN groups. To be used with PINN Dataloader. You can add groups of inputs representing boundary condition, 
    collocations points etc using the `add_group()` method.

    The `__getitem__` method returns a `PINN_dict` object where keys are group names. The items are a subgroup of the `PINN_group()` i.e. a `PINN_group()` containing a batch of the original inputs 
    '''
    def __init__(self,input_vars) -> None:
        super().__init__()
        self.groups:PINN_dict[str,PINN_group] = PINN_dict()
        self.input_vars = input_vars
    def add_group(self,name:str,inputs:Union[torch.Tensor,List,Tuple],targets:Union[torch.Tensor,None] = None,batch_size:int = 1,*,causal:bool = False,shuffle: bool = False,time_col:int = -1):
        '''
        Add group to dataset

        Inputs:
            - name: str Name of group
            - inputs: Tensor or List|Tuple inputs of group. This represents inputs to the network. For multiple inputs, use a tuple or list
            - batch_size: int size of batch size to use for that group. 
            - targets: Tensor or None. Target output that matches with the input. Use this for data driven conditions
            - shuffle: bool. Shuffles the data if true. Default is False

        If multiple inputs are provided then it is assumed that the first dim size is the same across all inputs
        '''
        self.groups[name] = PINN_group(name,inputs,batch_size,targets,input_vars=self.input_vars,shuffle=shuffle)

    def update_group(self,name,**kwargs):
        '''
        Update PINN_Group Attributes

        Required args:
            - name: str Name of group
        
        Optional Keywords (args found in `PINN_Group()`):
            - inputs: Tensor or List|Tuple inputs of group. This represents inputs to the network. For multiple inputs, use a tuple or list
            - batch_size: int size of batch size to use for that group. 
            - targets: Tensor or None. Target output that matches with the input. Use this for data driven conditions
            - shuffle: bool. Shuffles the data if true. Default is False

        If multiple inputs are provided then it is assumed that the first dim size is the same across all inputs
        '''
        assert name in self.groups.keys(), 'Could not find the Pinn Group. Perhaps you misspelt it?'

        group = self.groups[name]
        for attr,value in kwargs.items():
            assert hasattr(group,attr)
            setattr(group,attr,value)



    def __len__(self) -> int: 
        return max([len(group) for group in self.groups.values()])
    
    def __getitem__(self,idx:int) -> PINN_dict:
        return PINN_dict({group.name:group.subgroup(idx[group.name]) for group in self.groups.values() })

    def Sampler(self):
        '''
        Sampler to use for custom batch indexing
        '''
        return PINN_sampler(self.groups)

    def group_names(self):
        '''
        Return keys of groups
        '''
        return list(self.groups.keys())
    
    def add_time(self,time_type,time_interval:Union[list,tuple] = None,point:float = None,dim:int = -1):
        '''
        Add time column to each PINN_group. The time variable can also be set indivually for each group using `PINN_dataset[group_name].add_time(...)` using the same arguments.
        see `torch_DE.utils.sampling.add_time()` for more details on the implementation.

        input args:
        - time_type: str 
            - `random interval` randomly sample time points from variable `time_interval` of [a,b]
            - `random point` randomly sample a single time point inside the interval [a,b]. i.e. all points in a tensor will have the same time point
            - `single point` set a specific time point for all points in a tensor given by keyword arguement `point`
        - dim: int  = -1 the dimension to add the time column to. The time column is always set as the last column
        - point: float the time point to set the tensors with
        - time_interval: list | tuple: time interval (a,b) to sample from. Must not be None if used with `random interval` or `random point`
        
        '''

        for group_name in self.groups.keys():
            self.groups[group_name].add_time(time_type,time_interval =time_interval,point=point,dim=dim)


class PINN_sampler(Sampler):
    def __init__(self,groups:PINN_dict):
        '''
        Sampler Class for dataloader.

        Different Groups have different batch sizes and number of elements. This Sampler ensures the number of batches is the same across all groups.

        If the number of batches is less than the maximum number of batches, we add batches to it (in the form of repeated indexing) until they match

        '''
        self.groups = groups
        self.remainder_flag = False

    @staticmethod
    def make_indices(groups) ->Dict[str,torch.Tensor]:
        '''
        Create the indices for each group.
        '''
        indices = {}
        #Figure out Maximum number of batches
        num_batches = {group.name: len(group)//group.batch_size for group in groups.values()}
        max_batches = max(num_batches.values())

        for group in groups.values():
            repeats = max_batches//num_batches[group.name]
            remainder = max_batches % num_batches[group.name]
            idx = torch.randperm(len(group)).repeat(repeats) if group.shuffle else torch.arange(len(group)).repeat(repeats)
            #Randomly fill the last remainder if not equal insize            
            idx = torch.cat([idx,idx[:remainder*group.batch_size]])
            indices[group.name] = idx

        return indices

    def __len__(self) -> int:
        '''
        Calculates group with the most number of batches
        '''
        num_batches,num_remainders = zip(*[ (len(group)//group.batch_size,len(group) % group.batch_size) for group in self.groups.values()])
        max_batches,max_idx = max(num_batches), np.argmax(num_batches)
        if self.remainder_flag is False:
            # If not max group not divisible by batch size then raise warning
            has_remainder = num_remainders[max_idx]
            if has_remainder != 0:
                warnings.warn("The group with the most number of batches is not divisible by it's batch size. The last batch will be ignored. This warning will no longer appear")
            self.remainder_flag = True
        
        return max_batches
    
    def __iter__(self) -> Iterator[Dict[str,torch.Tensor]] :
        self.indices = self.make_indices(self.groups)
        for i in range(self.__len__()):
            batch = {group.name: self.indices[group.name][i*group.batch_size:(i+1)*group.batch_size] for group in self.groups.values() }
            yield batch


def PINN_Dataloader(dataset:PINN_dataset,**kwargs) -> DataLoader:
    '''
    Returns a native Pytorch Dataloader for PINN training in Torch_DE. Due to the way PINN dataset works the following keywords are not available for the DataLoader:

    - batch_size
    - shuffle,
    - sampler
    - batch_sampler
    - drop_last

    Inputs:
        - dataset: `PINN_dataset()` for dataloader
        - **kwargs: any sort of keywords for the Dataloader not found above
    Output:
        - Dataloader Object from Pytorch e.g torch.utils.data.Dataloader
    '''
    for kwarg in kwargs:
        if kwarg in ['batch_size','shuffle','sampler','batch_sampler','drop_last']:
            raise ValueError(f'Invalid Keyword: {kwarg} found. Note batch_size,sampler,batch_sampler, drop_last keywords cannot be used with PINN_Dataloader.')
    return DataLoader(dataset,batch_size=None,sampler= dataset.Sampler(),**kwargs)