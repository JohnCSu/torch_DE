import torch
from collections.abc import Iterable
from typing import Union,Dict,Callable,Tuple,List
from torch import Tensor
import torch.utils
import torch.utils.data
from torch_DE.utils.loss_weighting import GradNorm,Causal_weighting
from torch_DE.utils.data import PINN_dict,PINN_dataset,PINN_group

import pandas as pd


import inspect
from functools import wraps
from torch import Tensor,TensorType

def DE_func(func):
    '''
    Decorator to turn your function into form suitable for troch DE. This is primarily for functions defined explicitly using their input varible names
    e.g. Poisson2D(u_xx,u_zz,x,y,**kwargs). Important note that **kwargs keyword variable MUST be defined otherwise an error is raised. The kwargs will absorb any
    unused variables (e.g. u_x , u_z for the above equation)

    Otherwise functions must be defined using dictionaries as inputs in the form f(x,y) where  x is the dictionary containing the corresponding input data 
    and y is a dictionary containing output and derivatives of the network
    '''
    sig = inspect.signature(func)
    assert inspect.Parameter.VAR_KEYWORD in [param.kind for param in sig.parameters.values()] , 'To use residual decorator, you need to have the **kwargs declared in your function input e.g foo(x,y,**kwargs)'
    @wraps(func)
    def DE_func_wrapper(network_input:dict[str,TensorType],network_output:dict[str,TensorType]):
        return func(**network_input,**network_output )
    DE_func_wrapper.is_decorated = True
    return DE_func_wrapper



class Loss():
    def __init__(self,error_and_weights,power,causal,eps = 1):
        '''
        Stores the losses from the loss handler. To access values use the following methods:

        - `point_error()`       : Dict of Dict of Dict where `(str,(str,(str,Tensor))))`. Each Tensor is the results of a specific function applied to a group output e.g `loss['boundary']['noslip']['u']` contains the point error between the netowrk output u and the actual boundary value
        - `MSE()`               : Dict of Dictionaries where `(str,(str,(str,Tensor))))` here Tensor is a single element. Returns the MSE of each Tensor in point_error
        - `group_loss()`        : Dict of Dictionaries where `(str,(str,Tensor)))`. Returns the losses of each group under each loss term. e.g in 2D `loss['boundary']['no slip']` will be the sum of the loss constraints of both u and v 
        - 'individual_loss()'   : Dict of the form `(str,Tensor)` Returns the loss of each different loss type terms e.g. Residual, Boundary, Initial conditions
        - 'sum'                 : Tensor with 1 elem. Returns a single scalar value representing the objective function
        '''
        self.error_and_weights = error_and_weights
        
        self._point_error = None

        self._weighted_error = None
        self._MSE = None
        self._group_loss =None
        self._individual_loss = None
        self._sum = None
        self._power = power

        self.causal = causal
        self.eps = eps
        self.causal_weighting = None
        

    
    def num_losses(self):
        return len(self.point_error(flatten=True))

    # def __len__(self):
    #     '''
    #     Returns a 
    #     '''
    #     return list( [v for dict_1 in self.error_and_weights.values() for dict_2 in dict_1.values() for v in dict_2.values()] )
    
    def point_error(self,flatten = False) -> Dict[str,Dict[str,Dict[str,Tuple[torch.Tensor,torch.Tensor]]]]:
        '''
        Returns:
            Dict of Dict of Dict where `(str,(str,(str,Tensor))))`. Each Tensor is the results of a specific function applied to a group output \n 
            e.g `loss['boundary']['noslip']['u']` contains the point error between the netowrk output u and the actual boundary value

            if flatten is true then returns a list of of every tuple (weight,point_error tensor) in point error
        '''
        if self._point_error is None:
            self._point_error = {loss_type: {group_name : {term_name: point_error for term_name,(point_error,_) in group.items() } for (group_name,group) in (loss_group.items()) } for (loss_type,loss_group) in self.error_and_weights.items() }
        if flatten:
            return self.flatten(self._point_error,order = 2)
        
        return self._point_error

    def weighted_error(self,flatten = False)-> Dict[str,Dict[str,Dict[str,torch.Tensor]]]:
        '''
        Returns:
            Dict of Dict of Dict `Dict[str,Dict[str,Dict[str,Tuple[torch.Tensor,torch.Tensor]]]]`. Here we raise all tensor to a power (default 2) and then apply the spatial weighting.
            If `causal` was set to True, then we apply the causality weighting scheme

            if flatten is true then returns a list of of every tensor after weighting in weighted_error
        '''
        if self._weighted_error is None:
            self._weighted_error = {loss_type: {group_name : {term_name: weight*(point_error.pow(self._power)) for term_name,(point_error,weight) in group.items() } for (group_name,group) in (loss_group.items()) } for (loss_type,loss_group) in self.error_and_weights.items() }
            
            
            if self.causal:
                self.causal_weighting = Causal_weighting(self._weighted_error,eps = self.eps)
                self._weighted_error['residual'] = {group_name:{term_name: self.causal_weighting*term_loss for term_name,term_loss in group.items()} for group_name,group in self._weighted_error['residual'].items()}

        if flatten:
            return self.flatten(self._weighted_error,order = 2)

        return self._weighted_error
    
    def MSE(self,flatten = False,names = False) -> Dict[str,Dict[str,Dict[str,torch.Tensor]]]:
        '''
        Returns:
            Dict of Dictionaries where `(str,(str,(str,Tensor))))` here Tensor is a single element. Returns the MSE of each Tensor in weighted_error()

            if flatten is true then returns a list of of every tensor after the averaging has been applied
        '''
        if self._MSE is None:
            self._MSE  ={loss_type : {group_name:{term_name: point_loss.mean() for term_name,point_loss in group.items()} for group_name,group in loss_group.items() } for loss_type,loss_group in self.weighted_error().items() } 
        
        if flatten:
            return self.flatten(self._MSE,order = 2,names=names)

        return self._MSE
    
    def group_loss(self,flatten =False,names = False) -> Dict[str,Dict[str,torch.Tensor]]:
        '''
        Returns:
            Dict of Dictionaries where `(str,(str,Tensor)))`. Returns the losses of each group under each loss term.\n
            e.g in 2D fluid flow `loss['boundary']['no slip']` will be the sum of the loss constraints of both u and v

            if flatten is true then returns a list of of every tensor after summing 
        '''
        if self._group_loss is None:
            group_loss = {loss_type: {group_name:sum(group.values()) for group_name,group in loss_group.items() } for loss_type,loss_group in self.MSE().items() }
            self._group_loss = group_loss
        
        if flatten:
            return self.flatten(self._group_loss,order = 1,names=names)

        return self._group_loss
    
    def individual_loss(self,flatten =False,names = False) -> Dict[str,torch.Tensor]:
        '''
        Returns:
            Dict where `Dict[str,Tensor]` Returns the loss of each different loss type e.g. single value for Residual, Boundary, Initial conditions term

            if flatten is true then returns each individual loss as list form. Equivalent to `list(Loss.individual_loss().values())` 
            '''
        if self._individual_loss is None:
            individual_loss = {loss_type: sum(loss_group.values()) for loss_type,loss_group in self.group_loss().items()}
            self._individual_loss = individual_loss
        if flatten:
            return self.flatten(self._group_loss,order = 0,names=names)
        
        return self._individual_loss
    
    def sum(self) -> torch.Tensor:
        '''
        Returns:
            The sum of all terms into a single element tensor. Use before calling backwards()
        '''
        if self._sum is None:
            self._sum = sum(self.individual_loss().values())
        return self._sum

    @staticmethod
    def flatten(loss:Dict,order,names = False):
        if order == 0:
            #Dict[str,tensor]
            flattened_names = list(loss.keys())
            flattened_losses =  list(loss.values())
        if order == 1:
            #Dict[str,Dict[str,torch.Tensor]]:
            flattened_names = [f'{group_name}_{v}' for group_name,dict_1 in loss.items() for v in dict_1.keys()]
            flattened_losses = list([v for dict_1 in loss.values() for v in dict_1.values()])
        if order == 2:
            #Dict[str,Dict[str,Dict[str,torch.Tensor]]]
            flattened_names = list([f'{indi_name}_{group_name}_{v}' for indi_name,dict_1 in loss.items() for group_name,dict_2 in dict_1.items() for v in dict_2.keys()])
            flattened_losses = list([v for dict_1 in loss.values() for dict_2 in dict_1.values() for v in dict_2.values()])

        if names:
            return flattened_losses,flattened_names
        else:
            return flattened_losses 
    def print_losses(self,epoch,weights = None,summary = 'groups'):
        with torch.no_grad():
            print(f'Epoch {epoch} :--: Total Loss {float(self.sum()): .3E}  ',end = '  ')

            if summary == 'groups':
                for name,loss in self.individual_loss().items():
                    print( f'{name} Loss: {float(loss): .3E}',end = '  ')
                if self.causal_weighting is not None:
                    print(f'Causal Weighting Stats: Max: {float(self.causal_weighting.max()):.3E}, Mean: {float(self.causal_weighting.mean()):.3E}, Min: {float(self.causal_weighting.min()):.3E}',end = '   ')
                print()
            if summary == 'full':
                for name,loss in self.MSE(flatten=True):
                    print( f'{name} Loss: {float(loss): .3E}',end = '  ')


    

class Loss_handler():
    def __init__(self,dataset:PINN_dataset) -> None:
        '''
        Loss_handler is designed to work with PINN_dataholder and DE_Getter()

        '''
        self.update_dataset(dataset)        
        self.loss_groups = {}
        self.losses = None
        self.logger = None


        self.losses: pd.DataFrame = pd.DataFrame(columns = [  'group',
                                                'loss_type',
                                                'variable',
                                                'evaluation',
                                                'weighting',
                                                'residual',
                                                'MSE'])
                                                

    def update_dataset(self,dataset:PINN_dataset):
        assert isinstance(dataset,(PINN_dataset,torch.utils.data.Dataset))
        self.dataset = dataset
        self.groups = dataset.groups
        self.group_names = self.groups.keys()




    def __call__(self,group_input:Dict,group_output:Dict,**kwargs):
        return self.calculate(group_input,group_output,**kwargs)

    
    def __len__(self):
        return len(self.losses)

    def log_loss(self):
        losses = self.losses.individual_loss()
        if self.logger is None:
            self.logger = {loss_type : [] for loss_type in losses.keys()}
            self.logger['total'] = []
        with torch.no_grad():
            for loss_type,loss in losses.items():
                self.logger[loss_type].append(float(loss.cpu()))

            self.logger['total'].append(float(self.losses.sum().cpu()))


    def check_groups(self,dataset:PINN_dataset):
        '''
        Check that the groups specified are matched one-to-one with the PINN dataset, Otherwise raise an error
        '''

        pass

    def calculate(self,batched_input:dict[str,PINN_group],batched_output:dict[str,dict[str,Tensor]],power:int = 2)->Loss:
        # Dict[group,dict[loss_type,tensor]]
        error_func = lambda x,y: torch.abs(x-y).pow(power)


        for group in batched_input.keys():
            group_bool = self.losses['group'] == group
            
            group_output = batched_output[group]
            group_input = batched_input[group].inputs
            
            self.losses.loc[group_bool,'residual'] = self.losses.loc[group_bool,'evaluation'].apply(lambda func: func(group_input,group_output))
            
        
        return self.losses

    def calculate_point_error_and_weights(self,group_input:Dict,group_output:Dict,loss_group_first = True) -> Tuple[Dict[str,Dict[str,Dict[str,torch.Tensor]]],Dict[str,Dict[str,Dict[str,torch.Tensor]]]]:
        '''
        Create a nested Dict[str,Dict[str,Dict[str,Tensor]]] of both the point error and weights of each points.

        Returns a tuple pair of Dict[str,Dict[str,Dict[str,Tensor]]] with the first element being the point errors and second element/dict being the point weights for each term
        '''
        point_losses = {}

        for (loss_type,loss_group) in self.loss_groups.items():
                point_losses[loss_type] = {}
                for (group_name,group) in loss_group.items():
                    point_losses[loss_type][group_name] = {}
                    point_loss = point_losses[loss_type][group_name]
                    for loss_name,(loss_func,weight_func) in group.items():
                        point_loss[loss_name] = (loss_func(group_input,group_output),weight_func(group_input))
        
        return point_losses
    

    def add_custom_function(self,group,func_dict,weighting = 1):
        '''
        Add a custom function. 
        Inputs:
        - group: str - the name to place all custom functions in. This group can be arbitary. This is useful to group custom functions together. For example grouping a specific set of function under 'mass flow'
        - func_dict: dictionary: a Dictionary containing the function name and a tuple (func,kwargs) to put in group_name. The dictionary syntax of (key,value) --> (func_name,(function,kwargs))


        The function should take in two inputs f(x_dict,group_dict): 
        x_dic a dictionary where the each group contains a batch of input points to the network. For example x_dic['spam'] will return the the tensor of size (N,D) (e.g. x,y,z) from the group 'spam'. 

        group_dic a dictionary of dictionaries where the first set of keys return the group and the second set of keys return the output variable. For example group_dic['foo']['u'] will return the 'u' variables from the group foo.

        Note that if the loss handler is called or the function self.caluclate is called, the result will then be raised to a power (defaut 2) and then averaged.

        '''
        def f(kwargs):
            def inner_f(x,g):
                return func(x,g,**kwargs)
            return inner_f
        
        custom_dic = self.group_checker('custom',group)
        if not isinstance(weighting,dict):
            weighting = {func_name: weighting for func_name in func_dict.keys()}

        for weight,(func_name,func_items) in zip(weighting.values(),func_dict.items()):
            (func,kwargs) = func_items if len(func_items) == 2 else (func_items[0],{})
            custom_dic[group][func_name] = (f(kwargs),self.make_weighting_func(weight,group))


    def set_terms(self,loss_type,group,var_dict: dict[str,Union[float,Callable]], weighting: Union[float,dict,Callable]):
        # The output of DE_Getter is a dictionary [group][vars]

        #Get all the losses associated with the loss_type (e.g. boundary or IC) and group


        group_loss_type_df = self.losses[(self.losses['group'] == group)&(self.losses['loss_type'] == loss_type)]
        #We want to match up the weighting to the var_dict
        if not isinstance(weighting,dict):
            weighting = {var_comp: weighting for var_comp in var_dict.keys()}  
    
        for (var_name,evaluation_func),(weighting) in zip(var_dict.items(),weighting.values()):

            #First Check that varible not already added in loss type for group
            if var_name in group_loss_type_df['variable']:
                raise ValueError(f'variable name {var_name} already exists in group {group} as a {loss_type} term')
            
            weight_func = lambda group_input : weighting if not callable(weight_func) else weighting 
                
            loss = pd.DataFrame([{
                'loss_type': loss_type,
                'group': group,
                'variable': var_name,
                'evaluation': evaluation_func,
                'weighting': weight_func,
                'residual': None,
                'MSE': None
            }])

            self.losses = pd.concat([self.losses,loss])
    @staticmethod
    def create_residual_from_rhs(var_name,rhs):
        '''
        Given a right hand side (rhs), create a residual function such that var_name-rhs = 0
        '''
        
        rhs_func = lambda **kwargs: rhs if not callable(rhs) else rhs
        rhs_func = DE_func(rhs_func)
        
        #Residual Functions is group_dict[var_name] - rhs(x)
        return lambda group_input,group_output: group_output[var_name] - rhs_func(group_input,group_output)

    def add_boundary(self,group,bound_dict:dict[str,Union[float,Callable]],weighting:Union[float,Callable,Dict] = 1):
        bound_dict = {var_name:self.create_residual_from_rhs(var_name,rhs) for var_name,rhs in bound_dict.items()}
        self.set_terms('boundary',group,bound_dict,weighting)

    
    def add_initial_condition(self,group:str,ic_dict:dict,weighting = 1):
        ic_dict = {var_name:self.create_residual_from_rhs(var_name,rhs) for var_name,rhs in ic_dict.items()}
        self.set_terms('initial condition',group,ic_dict,weighting)
        
    def add_residual(self,group,residuals:dict[str,Callable],weighting = 1): 
        self.set_terms('residual',group,residuals,weighting)
        

    # def add_boundary(self,group,bound_dict,weighting:Union[float,Callable,Dict] = 1):
        
    #     self.set_terms('boundary','data_loss_func',group,bound_dict,weighting)
    
    # def add_data_constraint(self,group:str,data_keys:List[str] = None,weighting = 1):
    #     '''
    #     Add a data driven contstraint to a PINN. This is used when we have data is strictly tied to the input data i.e. data from a sensor over time. 
    #     '''
    #     # self.set_terms('data_driven','data_driven',group,data_keys,weighting)

    #     if self.groups[group].targets is None:
    #         raise ValueError(f'The group {group} does not have any target data attributed to it')

    #     if data_keys is None:
    #         data_keys = self.groups[group].targets.keys()

    #     loss_group = self.group_checker('data driven',group)
    #     if not isinstance(weighting,dict):
    #         weighting = {var_comp: weighting for var_comp in data_keys}
        

    #     for key,weight in zip(data_keys,weighting.values()):
    #         loss_group[group][key] = (self.data_driven_func(group,key), self.make_weighting_func(weight,group))
    
    # def add_residual(self,group,res_dict:dict,weighting = 1):
    #     self.set_terms('residual','residual_loss_func',group,res_dict,weighting)
    
    # def add_initial_condition(self,group:str,ic_dict:dict,weighting = 1):
    #     self.set_terms('initial condition','data_loss_func',group,ic_dict,weighting)
        
    # def add_periodic(self,group_1:str,group_2:str,var:str,weighting = 1):
    #     '''
    #     We treat group_1 as the main group and group_2 as the secondary so this is stored in the group_1

    #     The key structure is [group_1][group_2_var]
    #     '''
    #     loss_group = self.group_checker('periodic',group_1)
    #     assert group_2 in self.groups, "The group {group} does not exist in the loss handler. Please check your spelling"

    #     if isinstance(var,str):
    #         loss_group[group_1][f'{group_2}_{var}'] = (self.periodic_loss_func(group_1,group_2,var),self.make_weighting_func(weighting,group_1))
            
    #     else:
    #         raise TypeError(f'varaible var needs to be type string Instead found type f{type(var)}')

    

    def group_checker(self,loss_type,group) -> dict:
            '''
            Check if loss_type exists and if group in loss type exists. Otherwise add them
            '''
            if loss_type not in self.loss_groups.keys():
                self.loss_groups[loss_type] = {}
            if loss_type != 'custom':
                assert group in self.group_names, f"The group {group} does not exist in the loss handler. Please check your spelling"
            loss_group = self.loss_groups[loss_type]
            if group not in loss_group.keys():
                loss_group[group] = {}
            return loss_group



    @staticmethod
    def make_weighting_func(weighting,group_name):
        if callable(weighting): 
            return lambda group_input : weighting(group_input[group_name].inputs)
        else:
            return lambda group_input : weighting
        

    def periodic_loss_func(self,group_1,group_2,var):
        
        def periodic_loss(group_input,output_dict):
            return output_dict[group_1][var] - output_dict[group_2][var]
        
        return periodic_loss


    def data_driven_func(self,group,data_var):
        return lambda group_input,output_dict: (output_dict[group][data_var] - group_input[group].targets[data_var]) 

    def data_loss_func(self,group:str,var_comp:str,value_func):
        
        value_func2 =  (lambda x : value_func) if not callable(value_func) else value_func
        
        def data_loss_constructor(group_input,output_dict):
                '''
                u is a dictionary containing all the ouputs and derivatives of a
                '''
                return ((output_dict[group][var_comp] - value_func2(group_input[group].inputs)))
            

        return data_loss_constructor

    def residual_loss_func(self,group,res_name,res_func):
        '''
        Create a residual loss function
        res_func: function that takes in derivatives and outputs and potentially inputs. Should be of the form R(x,u,u_t,..., **kwargs) Must have the kwargs term
                Note that x here is an overall variable for spatial input. If you need to seperate the spatial input into individual components, e.g in to x,y,z
                Do so inside the function
        '''
        def residual_loss_constructor(group_input,output_dict):
                '''
                u is a dictionary containing all the ouputs and derivatives of a
                '''
                return res_func( x = group_input[group],**output_dict[group])

        return residual_loss_constructor
