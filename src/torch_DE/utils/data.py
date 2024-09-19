
import torch
from torch.utils.data import Dataset,DataLoader,Sampler
import numpy as np
import warnings
from typing import Dict,List,Tuple,Union,Iterator
import inspect
from torch_DE.utils.time import add_time
from torch_DE.symbols import Variable_dict
class PINN_group():
    def __init__(self,name,inputs,batch_size,targets:Union[torch.Tensor,dict] = None,*,shuffle = False,add_time_interval:List = None,causal = False,time_col = -1):
        '''
        Container for data for a defined grouped
        '''
        self.name = name
        self.inputs = inputs
        self.batch_size = batch_size

        if isinstance(targets,dict):
            self.targets = PINN_dict(targets)
        else:
            self.targets = PINN_dict({'target': targets}) if isinstance(targets,torch.Tensor) else None
        
        self.shuffle = shuffle 
        self.causal = causal
        self.time_col = time_col
        #Check
        self.has_multiple_inputs = False
        if isinstance(self.inputs,(list,tuple)):
            self.has_multiple_inputs = True
            #Check if first dim is same size
            assert set([x.shape[0] for x in self.inputs]) == 1 , 'The size of first dim must be the same across inputs'
            
        else:
            if not isinstance(self.inputs,torch.Tensor):
                raise TypeError('inputs can only be typr torch.Tensor, List or tuple!')

        assert batch_size <= self.__len__()

        self.num_elems = self.__len__()


        if self.has_multiple_inputs and self.causal:
            raise ValueError('Causal sorting currently does not support groups with multiple input tensors')

    def __len__(self):
        if self.has_multiple_inputs:
            return len(self.inputs[0].shape[0])
        else:
            return self.inputs.shape[0]

    def to(self,*args,**kwargs):
        if self.has_multiple_inputs:
            self.inputs = tuple(x.to(*args,**kwargs) for x in self.inputs)
        else:
            self.inputs=self.inputs.to(*args,**kwargs)
        if self.targets is not None:
            self.targets = self.targets.to(*args,**kwargs)
        return self
    def subgroup(self,idx):
        '''
        Given an list of indices, return a smaller PINN group containing data only on those indices. This is used for batching each group
        '''
        inputs = self.inputs[idx]
        
        if self.has_multiple_inputs:
            inputs = tuple(x[idx] for x in self.inputs)
        
        targets = None
        if self.targets is not None:
            targets = {target: target_tensor[idx] for target,target_tensor in self.targets.items()}
        
        if self.causal:
            #Find col corresponding to time column, sort asending order and return indices
            sorted_time_idx = inputs[:,self.time_col].sort()[1]
            inputs = inputs[sorted_time_idx]
            if self.targets is not None:
                for target in self.targets.keys():
                    self.targets[target] = self.targets[target][sorted_time_idx] 

        return PINN_group(self.name,inputs,self.batch_size,targets,causal = self.causal,shuffle=self.shuffle)


    def add_time(self,time_type,time_interval:Union[list,tuple] = None,point:float = None,dim:int = -1):
        if self.has_multiple_inputs:
            for tensor in self.inputs[1:]:
                assert tensor.shape == self.inputs[0].shape, 'tensors in each input need to be of same shape. You may need to manually assign time to each individual inputs'
            tensors = self.inputs
        else:
            tensors = [self.inputs]
        self.inputs = add_time(time_type,*tensors,time_interval =time_interval,point=point,dim=dim)
        


    def __getitem__(self,idx):
        return self.subgroup(idx)



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
    def __init__(self) -> None:
        super().__init__()
        self.groups:PINN_dict[str,PINN_group] = PINN_dict()

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
        self.groups[name] = PINN_group(name,inputs,batch_size,targets,causal=causal,shuffle=shuffle,time_col= time_col)

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
        num_batches = {group.name: group.num_elems//group.batch_size for group in groups.values()}
        max_batches = max(num_batches.values())

        for group in groups.values():
            repeats = max_batches//num_batches[group.name]
            remainder = max_batches % num_batches[group.name]
            idx = torch.randperm(group.num_elems).repeat(repeats) if group.shuffle else torch.arange(group.num_elems).repeat(repeats)
            #Randomly fill the last remainder if not equal insize            
            idx = torch.cat([idx,idx[:remainder*group.batch_size]])
            indices[group.name] = idx

        return indices

    def __len__(self) -> int:
        '''
        Calculates group with the most number of batches
        '''
        num_batches,num_remainders = zip(*[ (group.num_elems//group.batch_size,group.num_elems % group.batch_size) for group in self.groups.values()])
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