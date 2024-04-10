import torch
from collections.abc import Iterable
from typing import Union,Dict

class Loss():
    def __init__(self,point_error,power):
        '''
        Stores the losses from the loss handler. To access values use the following methods:

        - `point_error()`       : Dict of Dict of Dict where `(str,(str,(str,Tensor))))`. Each Tensor is the results of a specific function applied to a group output e.g `loss['boundary']['noslip']['u']` contains the point error between the netowrk output u and the actual boundary value
        - `MSE()`               : Dict of Dictionaries where `(str,(str,(str,Tensor))))` here Tensor is a single element. Returns the MSE of each Tensor in point_error
        - `group_loss()`        : Dict of Dictionaries where `(str,(str,Tensor)))`. Returns the losses of each group under each loss term. e.g in 2D `loss['boundary']['no slip']` will be the sum of the loss constraints of both u and v 
        - 'individual_loss()'   : Dict of the form `(str,Tensor)` Returns the loss of each different loss type terms e.g. Residual, Boundary, Initial conditions
        - 'sum'                 : Tensor with 1 elem. Returns a single scalar value representing the objective function
        '''
        
        self.point_error_ = point_error
        self.MSE_ = None
        self.group_loss_ =None
        self.individual_loss_ = None
        self.sum_ = None
        self.power_ = power

    def point_error(self) -> Dict[str,Dict[str,Dict[str,torch.Tensor]]]:
        '''
        Returns:
            Dict of Dict of Dict where `(str,(str,(str,Tensor))))`. Each Tensor is the results of a specific function applied to a group output \n 
            e.g `loss['boundary']['noslip']['u']` contains the point error between the netowrk output u and the actual boundary value
        '''
        return self.point_error_

    def MSE(self) -> Dict[str,Dict[str,Dict[str,torch.Tensor]]]:
        '''
        Returns:
            Dict of Dictionaries where `(str,(str,(str,Tensor))))` here Tensor is a single element. Returns the MSE of each Tensor in point_error()
        '''
        if self.MSE_ is None:
            MSE_terms = {}
            for loss_type,loss_group in self.point_error_.items():
                MSE_terms[loss_type] = {group_name:{term_name: point_loss.pow(self.power_).mean() for term_name,point_loss in group.items()} for group_name,group in loss_group.items() }
            self.MSE_ = MSE_terms 
        
        return self.MSE_
    
    def group_loss(self) -> Dict[str,Dict[str,torch.Tensor]]:
        '''
        Returns:
            Dict of Dictionaries where `(str,(str,Tensor)))`. Returns the losses of each group under each loss term.\n
            e.g in 2D fluid flow `loss['boundary']['no slip']` will be the sum of the loss constraints of both u and v 
        '''
        if self.group_loss_ is None:
            group_loss = {loss_type: {group_name:sum(group.values()) for group_name,group in loss_group.items() } for loss_type,loss_group in self.MSE().items() }
            self.group_loss_ = group_loss
        return self.group_loss_
    
    def individual_loss(self) -> Dict[str,torch.Tensor]:
        '''
        Returns:
            Dict where 1(str,Tensor)` Returns the loss of each different loss type e.g. single value for Residual, Boundary, Initial conditions term
        '''
        if self.individual_loss_ is None:
            individual_loss = {loss_type: sum(loss_group.values()) for loss_type,loss_group in self.group_loss().items()}
            self.individual_loss_ = individual_loss
    
        return self.individual_loss_
    
    def sum(self) -> torch.Tensor:
        '''
        Returns:
            The sum of all terms into a single element tensor. Use before calling backwards()
        '''
        if self.sum_ is None or power != self.power_:
            self.sum_ = sum(self.individual_loss().values())
        return self.sum_




class Loss_handler():
    def __init__(self,groups:Iterable) -> None:
        '''
        Loss_handler is designed to work with PINN_dataholder and DE_Getter()

        '''
        self.weighting = None
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
        with torch.no_grad():
            for loss_type,loss in losses.items():
                    self.logger[loss_type].append(float(loss.cpu()))
            
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
            - 'MSE'
            - 'group_loss'
            - 'individual_loss'
            - 'sum'

        Output:
            - Loss object . Call the following methods to get the following different outputs:
                - `point_error()`   : Dict of Dict of Dict where `(str,(str,(str,Tensor))))`. Each Tensor is the results of a specific function applied to a group output e.g `loss['boundary']['noslip']['u']` contains the point error between the netowrk output u and the actual boundary value
                - `MSE()`           : Dict of Dictionaries where `(str,(str,(str,Tensor))))` here Tensor is a single element. Returns the MSE of each Tensor in point_error
                - `group_loss()`    : Dict of Dictionaries where `(str,(str,Tensor)))`. Returns the losses of each group under each loss term. e.g in 2D Fluid Flow `loss['boundary']['no slip']` will be the sum of the loss constraints of both u and v 
                - `individual_loss()`    : Dict where 1(str,Tensor)` Returns the loss of each different loss type terms e.g. Residual, Boundary, Initial conditions
                - `sum()`           : Tensor with 1 elem. Returns a single scalar value representing the objective function
        '''

        point_losses = self.calculate_point_error(group_input,group_output)
        self.losses = Loss(point_losses,power)

        if output_type is None:
            return self.losses
        else:
            return getattr(self.losses,output_type)()


    def calculate_point_error(self,group_input,group_output):
        point_losses = {}
        for loss_type,loss_group in self.loss_groups.items():
                point_losses[loss_type] = {} 
                for group_name,group in loss_group.items():
                    point_losses[loss_type][group_name] = {}

                    point_loss = point_losses[loss_type][group_name]
                    
                    for loss_name,loss_func in group.items():
                        point_loss[loss_name] = loss_func(group_input,group_output)

        return point_losses
    


    def add_custom_function(self,group,func_dict):
        '''
        Add a custom function. 
        Inputs:
        - group: str - the name to place all custom functions in. This group can be arbitary. This is useful to group custom functions together. For example grouping a specific set of function under 'mass flow'
        - func_dict: dictionary: a Dictionary containing the function name and a tuple (func,kwargs) to put in group_name. The dictionary syntax of (key,value) --> (func_name,(function,kwargs))


        The function should take in two inputs f(x_dict,group_dict): 
        x_dic a dictionary where the  containing the input points to the network. For example x_dic['spam'] will return the input points (e.g. x,y,z) from the group 'spam'. 

        group_dic a dictionary of dictionaries where the first set of keys return the group and the second set of keys return the output variable. For example group_dic['foo']['u'] will return the 'u' variables from the group foo.

        Note that if the loss handler is called or the function self.caluclate is called, the result will then be raised to a power (defaut 2) and then averaged.

        No weighitng function is defined
        '''
        def f(kwargs):
            def inner_f(x,g):
                return func(x,g,**kwargs)
            return inner_f
        
        custom_dic = self.group_checker('Custom',group)
        for func_name,(func,kwargs) in func_dict.items():
            custom_dic[group][func_name] = f(kwargs)

        

    

    def add_boundary(self,group,bound_dict,weighting = 1):
        loss_group = self.group_checker('Boundary',group)
        for var_comp,value_func in bound_dict.items():
            loss_group[group][var_comp] = self.data_loss_func(group,var_comp,value_func,weighting)


    def add_residual(self,group,res_dict:dict,weighting = 1):
        loss_group = self.group_checker('Residual',group)
        for res_name,res_func in res_dict.items():
            loss_group[group][res_name] = self.residual_loss_func(group,res_func,weighting) 

    def add_initial_condition(self,group:str,ic_dict:dict,weighting = 1):
        loss_group = self.group_checker('Initial Condition',group)

        for var_comp,value_func in ic_dict.items():
            loss_group[group][var_comp] = self.data_loss_func(group,var_comp,value_func,weighting)


    def add_periodic(self,group_1:str,group_2:str,var:str,weighting = 1):
        '''
        We treat group_1 as the main group and group_2 as the secondary so this is stored in the group_1
        '''
        loss_group = self.group_checker('Periodic',group_1)
        assert group_2 in self.groups, "The group {group} does not exist in the loss handler. Please check your spelling"
        if isinstance(var,str):
            loss_group[group_1][group_2] = self.periodic_loss_func(group_1,group_2,var,weighting)
        elif isinstance(var,Iterable):
            for v in var:
                loss_group[group_1][group_2] = self.periodic_loss_func(group_1,group_2,v,weighting)
        else:
            raise TypeError(f'varaible var needs to be type string or iterable. Instead found type f{type(var)}')

    

    def group_checker(self,loss_type,group) -> dict:
            '''
            Check if loss_type exists and if group in loss type exists. Otherwise add them
            '''
            if loss_type not in self.loss_groups.keys():
                self.loss_groups[loss_type] = {}
            assert group in self.groups, "The group {group} does not exist in the loss handler. Please check your spelling"
            loss_group = self.loss_groups[loss_type]
            if group not in loss_group.keys():
                loss_group[group] = {}
            return loss_group



    @staticmethod
    def make_weighting_func(weighting):
        if callable(weighting): 
            return lambda x : weighting(x)
        else:
            return lambda x : weighting
        

    def periodic_loss_func(self,group_1,group_2,var,weighting=1):
        w_func = self.make_weighting_func(weighting)

        def periodic_loss(group_input,output_dict):
            return w_func(group_input[group_1])*((output_dict[group_1][var] - output_dict[group_2][var]))
        
        return periodic_loss


    def data_loss_func(self,group,var_comp:str,value_func,weighting = 1):
        w_func = self.make_weighting_func(weighting)        
        #Value_func is either a predefined set of data points (e.g. u = 0) or a function of space
        if callable(value_func):
            comp_func = lambda x : value_func(x)
        else:
            comp_func =  lambda x: value_func

        def data_loss_constructor(group_input,output_dict):
                '''
                u is a dictionary containing all the ouputs and derivatives of a
                '''
                return w_func(group_input[group])*((output_dict[group][var_comp] - comp_func(group_input[group])))
            

        return data_loss_constructor

    def residual_loss_func(self,group,res_func,weighting = 1):
        '''
        Create a residual loss function
        res_func: function that takes in derivatives and outputs and potentially inputs. Should be of the form R(x,u,u_t,..., **kwargs) Must have the kwargs term
                Note that x here is an overall variable for spatial input. If you need to seperate the spatial input into individual components, e.g in to x,y,z
                Do so inside the function
        '''

        if callable(weighting): 
            w_func = lambda x : weighting(x)
        else:
            w_func = lambda x : weighting
        

        def residual_loss_constructor(group_input,output_dict):
                '''
                u is a dictionary containing all the ouputs and derivatives of a
                '''
                return w_func(group_input[group])*((res_func( x = group_input[group],**output_dict[group])))

        return residual_loss_constructor
