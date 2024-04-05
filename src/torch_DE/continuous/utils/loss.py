import torch
from collections.abc import Iterable

class Loss_handler():
    def __init__(self,groups) -> None:
        '''
        Loss_handler is designed to work with PINN_dataholder and DE_Getter()

        '''
        self.weighting = None
        self.power = 2
        self.groups = set(groups)
        self.loss_groups = {loss_type : {} for loss_type in ['Boundary','Residual','Initial Condition','Periodic','Custom']}
        self.losses = {}
        self.logger = {loss_type : [] for loss_type in ['Boundary','Residual','Initial Condition','Periodic','Custom']}
    def __call__(self,group_input,group_output,power =2):
        return self.calculate_losses(group_input,group_output,power)

    def log_loss(self):
        with torch.no_grad():
            for loss_type,loss in zip(self.logger.keys(),self.losses.values()):
                self.logger[loss_type].append(float(loss.cpu()))
            

    def print_losses(self,i):
        print(f'Epoch {i} :--: ',end = '  ')
        for name,loss in self.losses.items():
            if loss != 0:
                print( f'{name} Loss: {float(loss): .3E}',end = '  ')
        print()

    def calculate(self,group_input,group_output,power = 2):
        '''
        We Group the losses based on group name together Need to update and be smarter
        '''
        point_losses = self.calculate_point_error(group_input,group_output)
        for loss_type,loss_group in point_losses.items():
            self.losses[loss_type] = sum([  sum([point_ls.pow(power).mean() for point_ls in group.values()]) for group in loss_group.values()])

        return self.losses


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
        custom_dic= self.loss_groups['Custom']
        if group not in custom_dic.keys():
            custom_dic[group] = {}
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

    def add_initial_condition(self,group:str,ic_dict:dict,weighting = 1):
        loss_group = self.group_checker('Initial Condition',group)

        for var_comp,value_func in ic_dict.items():
            loss_group[group][var_comp] = self.data_loss_func(group,var_comp,value_func,weighting)


    def group_checker(self,loss_type,group):
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
