import torch
from typing import Union,Dict,Iterable,Tuple,List
from torch_DE.utils.data import PINN_dict,PINN_group
from torch import Tensor
class engine():
    def __init__(self) -> None:
        self.derivatives = None
        self.net = None
        self.output_vars = None
    def __call__(self,*args,**kwargs):
        return self.calculate(*args,**kwargs)

    def dict_to_tensor(self,x: Union[torch.Tensor,PINN_dict[str,PINN_group]],exclude:str = None) -> Tuple[torch.Tensor,Union[None,List[str]],Union[None,List[int]]]:
        '''
        merges all the different tensors into one big concatenated tensor along batch dimension (assumes that first dimenstion is batch dimension)
        also creates a list of group names and sizes.

        if exclude is a str then that group is excluded (use if that excluded group is to be Differentiated)
        
        '''
        if isinstance(x,PINN_dict):
            exclude = [exclude] if isinstance(exclude,(str)) or exclude is None else exclude

            group_info:Tuple[Tuple[str],Tuple[Tensor],Tuple[str]] = zip(*[(group.name,group.inputs['input'],group.inputs['input'].shape[0]) for group in x.values() if group.name not in exclude])
            group_names,group_data,group_sizes =  group_info
            return torch.cat(group_data),group_names,group_sizes
        elif isinstance(x,torch.Tensor):
            return x,None,None
        else:
            raise ValueError(f'input should be of type dict or torch.tensor. Got instead {type(x)}')
    

    def tensor_to_dict(self,x,group_names,group_sizes):
        output = {}
        start_idx = 0
        for group_name,size in zip(group_names,group_sizes):
            output[group_name:x[start_idx:start_idx+size]]
            start_idx += size

        return output
    def net_pass_from_dict(self,x_dict,exclude = None)->  Dict[str,Dict[str,torch.Tensor]]:
        '''
        Allow passing a dict like object containing tensors to a network and output the results into a dictionary with the same keys as the input
        '''
        x,group_names,group_sizes = self.dict_to_tensor(x_dict,exclude=exclude)
        u = self.net(x)
        #We need to put this back into dictionary format
        output = {}
        start_idx = 0
        for group_name,size in zip(group_names,group_sizes):
            output[group_name] = {output_var: u[start_idx:start_idx+size,i] for output_var,i in self.output_vars.items()}
            start_idx += size
        
        return output
    @staticmethod
    def get_output_vars(derivatives:dict):
        return {output_var: idx[0] for output_var,idx in derivatives.items() if output_var.split('_')[0] == output_var}


    def calculate(self,x,**kwargs):
        pass
    
    def find_highest_order(self,derivatives):
        #Checking if variables in each derivative have been defined
        highest_order = 0
        for deriv in derivatives.keys():
            if deriv.find('_') != -1:
                _, indep_vars = deriv.split('_')
                #Order Function
                order = len(indep_vars)
                if order > highest_order:
                    highest_order = order  
        return highest_order
    

    def aux_function(self,aux_func,is_aux = True) -> object:
        '''
        Use for functorch functions to we can get the differentiated output as well as the network evaluation
        '''
    #aux_func has the output of (df,f) so we need it to output (df,(df,f))
    
        def initial_aux_func(x:torch.tensor) -> tuple[torch.tensor,torch.tensor]:
            out = aux_func(x)
            return (out,out)
        
        def inner_aux_func(x:torch.tensor) -> tuple[torch.tensor,torch.tensor]:
            out = aux_func(x)
            return (out[0],out)
        
        if is_aux:
            return inner_aux_func
        else:
            return initial_aux_func
        
        
