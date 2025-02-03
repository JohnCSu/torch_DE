import torch
from collections.abc import Iterable
from typing import Union,Dict,Callable,Tuple,List
from torch import Tensor
import torch.utils
import torch.utils.data
from torch_DE.utils.loss_weighting import GradNorm,Causal_weighting
from torch_DE.utils.data import PINN_dict,PINN_dataset,PINN_group
from torch_DE.equations.de_func import DE_func
import pandas as pd


import inspect
from functools import wraps
from torch import Tensor,TensorType


class Loss():
    def __init__(self,loss_df:pd.DataFrame,aggregation_method:str = 'mean',power:Union[int, None] = 2):
        '''
        Stores the losses from the loss handler. Provides additional functionality and variables to help calculate variables.
        '''
        self.losses = loss_df
        self.aggregation_method = aggregation_method
        self.error_func = lambda x: torch.abs(x).pow(power) if power is not None else lambda x: x
        self.aggregation = torch.mean if aggregation_method == 'mean' else torch.sum
    

        self.point_error_ = None
        self.weighted_point_error_ = None
        self.aggregated_loss_ = None
        
    def point_error(self) -> pd.Series:
        '''
        Apply the error function/raise the resiudal to a power.
        '''
        if self.point_error_ is None:
            self.point_error_ = self.losses['residual'].apply(self.error_func)
            self.losses['point_error'] = self.point_error_

        return self.point_error_
    
    def weighted_point_error(self) -> pd.Series:
        '''
        Apply Weighting to point error
        '''
        if self.weighted_point_error_ is None:
            self.weighted_point_error_ = self.losses['weighting']*self.point_error()
            self.losses['weighted_error'] = self.weighted_point_error_
        
        return self.weighted_point_error_
    def aggregated_loss(self) -> pd.Series:
        '''
        Aggregate each of the losses in each group to get the error for the (group,loss_type) combo
        '''
        if self.aggregated_loss_ is None:
            self.aggregated_loss_ = self.weighted_point_error().apply(self.aggregation)
            self.losses['aggregated_loss'] = self.aggregated_loss_
        return self.aggregated_loss_
    def sum(self) -> Tensor : 
        '''
        Sum up all aggregate losses to get the total loss
        '''
        return sum(self.aggregated_loss())
    
    def individual_losses(self):
        '''
        Returns all the losses as a Tensor
        '''
        return torch.stack(list(self.aggregated_loss()))

    def grouped_losses(self, groupby:str) -> pd.Series:
        '''
        Group up the losses based on the string input groupby. Uses `pd.DataFrame().groupby()` to achieve this.

        Valid strings are:
            - loss_type
            - group
            - variable
        '''
        self.aggregated_loss()
        if groupby in self.losses.columns:
            return self.losses.groupby(groupby)['aggregated_loss'].sum()
        raise ValueError(f'groupby must be one of the following strings: "loss_type", "group" or "variable". Got {groupby} instead')
    def backward(self):
        '''
        Calculate total loss and then Call the Backward method. Syntatic sugar
        '''
        self.sum().backward()

    def get_DataFrame(self):
        return self.losses
    
    def __len__(self):
        return len(self.losses)

    def print_Styled_DataFrame(self):
        '''
        Display HTML Dataframe format
        '''
        def tensor_shape_formatter(x):
            if isinstance(x, torch.Tensor):
                if len(x.shape) == 1 and x.shape[0] == 1:
                    return f'{float(x):.3E}'
                else:
                    return f'{x.shape}, device = {x.device}'  # Convert shape to string for display
            return x  # Leave other elements unchanged

    # Apply the custom formatting function
        return self.losses.style.format(tensor_shape_formatter)

    def print_losses(self,epoch,groupby = 'loss_type'):
        total_loss = self.sum()
        if groupby in self.losses.columns:
            losses = self.losses.groupby(groupby)['aggregated_loss'].sum()
            names = losses.index
            
            
        elif isinstance(groupby,None):
            names = list(self.losses['loss_type'] + '__' + self.losses['group'] + '__' + self.losses['variable'])
            losses = self.losses['aggregated_loss']
        else:
            raise ValueError(f'groupby must be either None or string and of the following strings: loss_type, group or variable. Got {groupby} instead')
        loss_strings = '\t'.join([f'{name}: {float(loss):.3E}' for name,loss in zip(names,losses)])

        print(f'Epoch {epoch}:- Total Loss {total_loss:.3E}\t {loss_strings}')






class Loss_handler():
    def __init__(self,dataset:PINN_dataset) -> None:
        '''
        Loss_handler is designed to work with PINN_dataholder and DE_Getter()

        '''
        self.update_dataset(dataset)        
        self.loss_groups = {}
        self.losses = None
        self.logger = None

        self.df_keys = [
                'loss_type',
                'group',
                'variable',
                'evaluation',
                'weighting_function',
                'weighting',
                'residual',
                'point_error',
                'weighted_error',
                'aggregated_loss',
                'custom',
                'group_input',
                'group_output']


        self.losses: pd.DataFrame = pd.DataFrame(columns = self.df_keys)
                                                

    def update_dataset(self,dataset:PINN_dataset):
        assert isinstance(dataset,(PINN_dataset,torch.utils.data.Dataset))
        self.dataset = dataset
        self.groups = dataset.groups
        self.group_names = self.groups.keys()




    def __call__(self,group_input:Dict,group_output:Dict,**kwargs) -> Loss:
        return self.calculate(group_input,group_output,**kwargs)

    
    def __len__(self):
        return len(self.losses)


    def check_groups(self,dataset:PINN_dataset):
        '''
        Check that the groups specified are matched one-to-one with the PINN dataset, Otherwise raise an error
        '''

        pass

    def calculate(self,batched_input:dict[str,PINN_group],batched_output:dict[str,dict[str,Tensor]],power:int = 2,aggregation_method = 'mean')->Loss:
        '''
        Calculate the residuals for each group. Summation and squaring the residuals is done in the loss object itself
        '''
        
        for group in batched_input.keys():
            group_bool = (self.losses['group'] == group) & (self.losses['custom'] == False)
            
            group_output = batched_output[group]
            group_input = batched_input[group]
            
            self.losses.loc[group_bool,'residual'] = self.losses.loc[group_bool,'evaluation'].apply(lambda func: func(group_input,group_output))
            self.losses.loc[group_bool,'weighting'] = self.losses.loc[group_bool,'weighting_function'].apply(lambda func: func(group_input,group_output))

        #All Custom take in the same 
        custom_funcs = self.losses['custom'] == True
        self.losses.loc[custom_funcs,'residual'] = self.losses.loc[custom_funcs,'evaluation'].apply(lambda func: func(batched_input,batched_output))
        self.losses.loc[custom_funcs,'weighting'] = 1.

        return Loss(self.losses,aggregation_method,power)



    def set_terms(self,loss_type,group,var_dict: dict[str,Union[float,Callable]], weighting: Union[float,dict,Callable],custom:bool = False):
        # The output of DE_Getter is a dictionary [group][vars]

        #Get all the losses associated with the loss_type (e.g. boundary or IC) and group

        assert group in self.dataset.group_names()
        group_loss_type_df = self.losses[(self.losses['group'] == group)&(self.losses['loss_type'] == loss_type)]
        #We want to match up the weighting to the var_dict
        if not isinstance(weighting,dict):
            weighting = {var_comp: weighting for var_comp in var_dict.keys()}  
    
        for (var_name,evaluation_func),(weighting_value) in zip(var_dict.items(),weighting.values()):

            #First Check that varible not already added in loss type for group
            if var_name in group_loss_type_df['variable']:
                raise ValueError(f'variable name {var_name} already exists in group {group} as a {loss_type} term')
            
            weight_func = (lambda group_input,group_output: weighting_value) if not callable(weighting_value) else weighting_value 
            

            loss_dict = dict.fromkeys(self.df_keys,None)
            loss_dict.update({
                'loss_type': loss_type,
                'group': group,
                'variable': var_name,
                'evaluation': evaluation_func,
                'weighting_function': weight_func,
                'custom': custom
            })
            
            loss = pd.DataFrame([loss_dict])

            self.losses = pd.concat([self.losses,loss],ignore_index= True)
    @staticmethod
    def create_residual_from_rhs(var_name,rhs):
        '''
        Given a right hand side (rhs), create a residual function such that var_name-rhs = 0
        '''
        rhs_func = DE_func(lambda **kwargs: rhs) if not callable(rhs) else rhs
        return lambda group_input,group_output: group_output[var_name] - rhs_func(group_input,group_output)



    def add_boundary(self,group,bound_dict:dict[str,Union[float,Callable]],weighting:Union[float,Callable,Dict] = 1):
        bound_dict = {var_name:self.create_residual_from_rhs(var_name,rhs) for var_name,rhs in bound_dict.items()}
        self.set_terms('boundary',group,bound_dict,weighting)

    
    def add_initial_condition(self,group:str,ic_dict:dict,weighting = 1):
        ic_dict = {var_name:self.create_residual_from_rhs(var_name,rhs) for var_name,rhs in ic_dict.items()}
        self.set_terms('initial condition',group,ic_dict,weighting)
        
    def add_residual(self,group,residuals:dict[str,Callable],weighting = 1): 
        self.set_terms('residual',group,residuals,weighting)
    

    def add_data_constraint(self,loss_type,group,data_vars: list[str],weighting = 1.):
        
        def data_driven_func(var_name):
            def data_driven_func_wrapper(group_input:PINN_group,group_output:PINN_dict):
                return group_output[var_name] - group_input.batchables[var_name]
            return data_driven_func_wrapper
        
        
        data_dict = {var_name: data_driven_func(var_name) for var_name in data_vars}

        
        self.set_terms(loss_type,group,data_dict,weighting)

    def add_periodic(self,group_1:str,group_2:str,variable:str):
        '''
        Add Periodic Conditions

        As this counts as a custom function (as it requires data from different groups), weighting point wise cannot be defined interms of input and output.

        '''

        def periodic(batched_input,batched_output):
            return batched_output[group_1][variable] - batched_output[group_2][variable]

        periodic_dict = {variable: periodic}    
        self.set_terms('periodic',group_1,periodic_dict,weighting=1.,custom=True)
    

    def add_custom_function(self,loss_type:str,group:str,func_dict:dict[str,Callable]):
        '''
        Add a custom function. 
        Inputs:
        - group: str - the name to place all custom functions in. This group can be arbitary. This is useful to group custom functions together. For example grouping a specific set of function under 'mass flow'
        - func_dict: dictionary: a Dictionary containing the function name and a tuple (func,kwargs) to put in group_name. The dictionary syntax of (key,value) --> (func_name,(function,kwargs))

        Note that weighing is not provided for custom functions. The weighting must be defined within the function itself.

        The function should take in two inputs f(batched_input,batched_output): 
        batched_input a dictionary where the each group contains a batch of input points to the network. For example `batched_input['spam']` will return a dictionary with keys `('inputs',*coords)` where inputs contains the full NxD tensor
        of the input and *coords are strings representing the input variable and slices of the full Tensor. e.g. 'x' would be equivalent to the slice of `batched_input['spam']['inputs'][:,0]`

        group_dic a dictionary of dictionaries where the first set of keys return the group and the second set of keys return the output variable. For example group_dic['foo']['u'] will return the 'u' variables from the group foo.

        Note that if the loss handler is called or the function self.caluclate is called, the result will then be raised to a power (defaut 2) and then averaged.

        '''
        custom = True
        self.set_terms(loss_type,group,func_dict,weighting=1.,custom=custom)

    