import torch
from collections.abc import Iterable
from typing import Union,Dict,Callable,Tuple

class Loss():
    def __init__(self,point_error,weighting,power):
        '''
        Stores the losses from the loss handler. To access values use the following methods:

        - `point_error()`       : Dict of Dict of Dict where `(str,(str,(str,Tensor))))`. Each Tensor is the results of a specific function applied to a group output e.g `loss['boundary']['noslip']['u']` contains the point error between the netowrk output u and the actual boundary value
        - `MSE()`               : Dict of Dictionaries where `(str,(str,(str,Tensor))))` here Tensor is a single element. Returns the MSE of each Tensor in point_error
        - `group_loss()`        : Dict of Dictionaries where `(str,(str,Tensor)))`. Returns the losses of each group under each loss term. e.g in 2D `loss['boundary']['no slip']` will be the sum of the loss constraints of both u and v 
        - 'individual_loss()'   : Dict of the form `(str,Tensor)` Returns the loss of each different loss type terms e.g. Residual, Boundary, Initial conditions
        - 'sum'                 : Tensor with 1 elem. Returns a single scalar value representing the objective function
        '''
        
        self._point_error = point_error
        self.weighting = weighting

        self._weighted_error = None
        self._MSE = None
        self._group_loss =None
        self._individual_loss = None
        self._sum = None
        self._power = power

    def point_error(self) -> Dict[str,Dict[str,Dict[str,torch.Tensor]]]:
        '''
        Returns:
            Dict of Dict of Dict where `(str,(str,(str,Tensor))))`. Each Tensor is the results of a specific function applied to a group output \n 
            e.g `loss['boundary']['noslip']['u']` contains the point error between the netowrk output u and the actual boundary value
        '''
        return self._point_error

    def weighted_error(self)-> Dict[str,Dict[str,Dict[str,torch.Tensor]]]:
        if self._weighted_error is None:
            self._weighted_error = {loss_type: {group_name : {term_name: point_error*weight for (term_name,point_error),weight in zip(group.items(),weight_group.values()) } for (group_name,group),weight_group in zip(loss_group.items(),weight_type.values()) } for (loss_type,loss_group),weight_type in zip(self._point_error.items(),self.weighting.values()) }
        return self._weighted_error
    
    def MSE(self) -> Dict[str,Dict[str,Dict[str,torch.Tensor]]]:
        '''
        Returns:
            Dict of Dictionaries where `(str,(str,(str,Tensor))))` here Tensor is a single element. Returns the MSE of each Tensor in point_error()
        '''
        if self._MSE is None:
            self._MSE  ={loss_type : {group_name:{term_name: point_loss.pow(self._power).mean() for term_name,point_loss in group.items()} for group_name,group in loss_group.items() } for loss_type,loss_group in self.weighted_error().items() } 
            
        return self._MSE
    
    def group_loss(self) -> Dict[str,Dict[str,torch.Tensor]]:
        '''
        Returns:
            Dict of Dictionaries where `(str,(str,Tensor)))`. Returns the losses of each group under each loss term.\n
            e.g in 2D fluid flow `loss['boundary']['no slip']` will be the sum of the loss constraints of both u and v 
        '''
        if self._group_loss is None:
            group_loss = {loss_type: {group_name:sum(group.values()) for group_name,group in loss_group.items() } for loss_type,loss_group in self.MSE().items() }
            self._group_loss = group_loss
        return self._group_loss
    
    def individual_loss(self) -> Dict[str,torch.Tensor]:
        '''
        Returns:
            Dict where 1(str,Tensor)` Returns the loss of each different loss type e.g. single value for Residual, Boundary, Initial conditions term
        '''
        if self._individual_loss is None:
            individual_loss = {loss_type: sum(loss_group.values()) for loss_type,loss_group in self.group_loss().items()}
            self._individual_loss = individual_loss
    
        return self._individual_loss
    
    def sum(self) -> torch.Tensor:
        '''
        Returns:
            The sum of all terms into a single element tensor. Use before calling backwards()
        '''
        if self._sum is None:
            self._sum = sum(self.individual_loss().values())
        return self._sum




class Loss_handler():
    def __init__(self,groups:Iterable) -> None:
        '''
        Loss_handler is designed to work with PINN_dataholder and DE_Getter()

        '''
        self.weighting = {}
        self.power = 2
        self.groups = set(groups)
        self.loss_groups = {}
        self.losses = None
        self.logger = None
    def __call__(self,group_input:Dict,group_output:Dict,power:int = 2,output_type:str = 'sum'):
        return self.calculate(group_input,group_output,power,output_type)

    def log_loss(self):
        losses = self.losses.individual_loss()
        if self.logger is None:
            self.logger = {loss_type : [] for loss_type in losses.keys()}
            self.logger['total'] = []
        with torch.no_grad():
            for loss_type,loss in losses.items():
                self.logger[loss_type].append(float(loss.cpu()))

            self.logger['total'].append(float(self.losses.sum().cpu()))


    def print_losses(self,epoch):
        print(f'Epoch {epoch} :--: ',end = '  ')
        for name,loss in self.losses.individual_loss().items():
            print( f'{name} Loss: {float(loss): .3E}',end = '  ')
        print()

    def calculate(self,group_input:Dict,group_output:Dict,power:int = 2,output_type:str=None)->Loss:
        '''
        Calculate losses. In Loss Handler we store terms in a 3-nested dictionary so accessing indiviudual terms is loss_type -> group__name -> var_name -> Output
        Inputs:
            - group_input: input training data. Should be of instance data-_handler or dict like
            - group_output: output network evaluation from DE_Getter. Should be dict like
            - power: power to raise loss terms by. Default is 2
            - output_type: None or str (default: 'sum'): call a specific loss method. if None return the loss object. Default 'sum' returns a single scalar value representing the objective function \n

            `None` returns the loss object while the below strings call the respective methods below:
            - 'point_error'
            - 'weighted_error'
            - 'MSE'
            - 'group_loss'
            - 'individual_loss'
            - 'sum'

        Output:
            - Loss object . Call the following methods to get the following different outputs:
                - `point_error()`   : Dict of Dict of Dict where `(str,(str,(str,Tensor))))`. Each Tensor is the results of a specific function applied to a group output e.g `loss['boundary']['noslip']['u']` contains the point error between the netowrk output u and the actual boundary value
                - `weighted_error()`: Dict of Dict of Dict where `(str,(str,(str,Tensor))))`. Each Tensor is the results of a specific function applied to a group output e.g `loss['boundary']['noslip']['u']` is the weighter point error between the netowrk output u and the actual boundary value
                - `MSE()`           : Dict of Dictionaries where `(str,(str,(str,Tensor))))` Here Tensor is a single element scalar. Returns the MSE of each Tensor across the batch dimension in weighted error
                - `group_loss()`    : Dict of Dictionaries where `(str,(str,Tensor)))`. Returns the losses of each group under each loss term. e.g in 2D Fluid Flow `loss['boundary']['no slip']` will be the sum of the loss constraints of both u and v 
                - `individual_loss()`    : Dict where 1(str,Tensor)` Returns the loss of each different loss type terms e.g. Residual, Boundary, Initial conditions
                - `sum()`           : Tensor with 1 elem. Returns a single scalar value representing the objective function
        '''

        point_losses,weighting = self.calculate_point_error(group_input,group_output)
        self.losses = Loss(point_losses,weighting,power)

        if output_type is None:
            return self.losses
        else:
            return getattr(self.losses,output_type)()


    def calculate_point_error(self,group_input:Dict,group_output:Dict) -> Tuple[Dict[str,Dict[str,Dict[str,torch.Tensor]]],Dict[str,Dict[str,Dict[str,torch.Tensor]]]]:
        '''
        Create a nested Dict[str,Dict[str,Dict[str,Tensor]]] of both the point error and weights of each points.

        Returns a tuple pair of Dict[str,Dict[str,Dict[str,Tensor]]] with the first element being the point errors and second element/dict being the point weights for each term
        '''
        point_losses = {}
        weight_dict = {}

         
        for (loss_type,loss_group),weight_group in zip(self.loss_groups.items(),self.weighting.values()):
                point_losses[loss_type] = {}
                weight_dict[loss_type] = {} 

                for (group_name,group),weight_g in zip(loss_group.items(),weight_group.values()):
                    point_losses[loss_type][group_name] = {}
                    weight_dict[loss_type][group_name] = {}

                    weight = weight_dict[loss_type][group_name]
                    point_loss = point_losses[loss_type][group_name]
                    
                
                    for (loss_name,loss_func),weight_func in zip(group.items(),weight_g.values()):
                        point_loss[loss_name] = loss_func(group_input,group_output)
                        weight[loss_name] = weight_func(group_input)
                        
                            

        return point_losses,weight_dict
    

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
        
        custom_dic,weight_dict = self.group_checker('Custom',group)
        if not isinstance(weighting,dict):
            weighting = {func_name: weighting for func_name in func_dict.keys()}

        for weight,(func_name,func_items) in zip(weighting.values(),func_dict.items()):
            (func,kwargs) = func_items if len(func_items) == 2 else func_items[0],{}
            custom_dic[group][func_name] = f(kwargs)
            weight_dict[group][func_name] = self.make_weighting_func(weight,group)


    def set_terms(self,loss_type,loss_type_func,group,var_dict,weighting):
        
        loss_group,weight_group = self.group_checker(loss_type,group)

        if not isinstance(weighting,dict):
            weighting = {var_comp: weighting for var_comp in var_dict.keys()}


        for weight,(var_comp,value_func) in zip(weighting.values(),var_dict.items()):
            loss_group[group][var_comp] = getattr(self,loss_type_func)(group,var_comp,value_func)
            weight_group[group][var_comp] =  self.make_weighting_func(weight,group)

    

    def add_boundary(self,group,bound_dict,weighting:Union[float,Callable,Dict] = 1):
        
        self.set_terms('Boundary','data_loss_func',group,bound_dict,weighting)
        

    def add_residual(self,group,res_dict:dict,weighting = 1):
        self.set_terms('Residual','residual_loss_func',group,res_dict,weighting)
    
    def add_initial_condition(self,group:str,ic_dict:dict,weighting = 1):
        self.set_terms('Initial Condition','data_loss_func',group,ic_dict,weighting)
        
    def add_periodic(self,group_1:str,group_2:str,var:str,weighting = 1):
        '''
        We treat group_1 as the main group and group_2 as the secondary so this is stored in the group_1

        The key structure is [group_1][group_2_var]
        '''
        loss_group,weight_group = self.group_checker('Periodic',group_1)
        assert group_2 in self.groups, "The group {group} does not exist in the loss handler. Please check your spelling"
        
        if group_1 not in self.weighting.keys():
            self.weighting[group_1] = {}

        if isinstance(var,str):
            loss_group[group_1][f'{group_2}_{var}'] = self.periodic_loss_func(group_1,group_2,var)
            weight_group[group_1][f'{group_2}_{var}'] =  self.make_weighting_func(weighting,group_1)
        else:
            raise TypeError(f'varaible var needs to be type string Instead found type f{type(var)}')

    

    def group_checker(self,loss_type,group) -> dict:
            '''
            Check if loss_type exists and if group in loss type exists. Otherwise add them
            '''
            if loss_type not in self.loss_groups.keys():
                self.loss_groups[loss_type] = {}
                self.weighting[loss_type] = {}
            assert group in self.groups, "The group {group} does not exist in the loss handler. Please check your spelling"
            loss_group = self.loss_groups[loss_type]
            weight_dict = self.weighting[loss_type]
            if group not in loss_group.keys():
                loss_group[group] = {}
                weight_dict[group] = {}
            return loss_group,weight_dict



    @staticmethod
    def make_weighting_func(weighting,group_name):
        if callable(weighting): 
            return lambda group_input : weighting(group_input[group_name])
        else:
            return lambda group_input : weighting
        

    def periodic_loss_func(self,group_1,group_2,var):
        
        def periodic_loss(group_input,output_dict):
            return output_dict[group_1][var] - output_dict[group_2][var]
        
        return periodic_loss


    def data_loss_func(self,group,var_comp:str,value_func):
        
        value_func2 =  (lambda x : value_func) if not callable(value_func) else value_func
        
        def data_loss_constructor(group_input,output_dict):
                '''
                u is a dictionary containing all the ouputs and derivatives of a
                '''
                return ((output_dict[group][var_comp] - value_func2(group_input[group])))
            

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
