import torch
from typing  import Dict,Tuple,List,Union,Callable
from matplotlib import pyplot as plt
from pandas import DataFrame
import pandas as pd
from torch_DE.post import Plotter,Tracker
from torch_DE.continuous import DE_Getter
from torch import nn,Tensor

class Evaluation(Plotter):
    '''
    Module to track and log errors and images. Inherits from `Plotter()` class and uses the `Tracker()` class via `self.logger`.

    This class implements some additional methods to help with plotting. For more details on kwargs see `Plotter()` and `Tracker()` classes

    '''
    def __init__(self,input_vars:Union[List,Tuple],output_vars:Union[List,Tuple],*,domain = None,eval_points:Tensor = None,time_int:List[float] = None,time_col = -1):
        '''
        inputs:
            - input_vars: list or tuple of strings representing input/independent variables
            - output_vars: list or tuple of strings representing output/dependent variables

            Keywords:
            - domain: `Domain2D()` object to generate evaluation points from. Mutually exclusive with eval_points
            - eval_points: `Tensor`, User supplied evaluation points to use. Must be of shape (N,D) where D is the number of input variables
            - time_int: `List[float,float]` . If the PINN is 2D spatial AND time-dependent then specify the initial and final time interval. None implies steady state PINN
            - time_col: `int` index of the time column in the (N,D) eval points tensor. default is -1
        '''
        
        super().__init__(input_vars,output_vars)

        # self.input_vars :Union[List,Tuple]   = input_vars
        # self.output_vars:Union[List,Tuple]  = output_vars
        self.ground_truth       :Dict[str,Tuple[Tensor,Tensor]] = {}

        self.logger = Tracker()
        
        #domain and eval_points must be XOR in regards to being Nonetype
        assert not (domain is None and eval_points is None), 'domain or eval points can\'t both be Nonetype!'
        assert not (domain is not None and eval_points is not None), 'only one of inputs domain and eval_points can be non Nonetype'

        self.domain = domain
        if self.domain is not None:
            self.set_contour_points(self.domain,resolution=100,time_interval = time_int,time_col=time_col) 
        elif eval_points is None:
            assert eval_points.shape[1] == len(input_vars), 'eval_points second dimension size must be same as number of input vars'
            self.contour_points = eval_points
        
        self.xs = {input_var: self.contour_points[:,i] for i,input_var in enumerate(input_vars)}


    def plot_residual(self,PINN:DE_Getter,equation:Callable,input_vars:List[str],time_point:float = None,eq_name:str = None,**kwargs):
        '''
        Plot contour of a residual of a PINN.

        inputs:
            - PINN: Some class or function such as `DE_Getter()` that return a dict of dicts where the first key are different groups (e.g. BC, ICs and interior points) 
            and the second keys are the outputs and derivatives of each respective group
            - equation: Callable function that calculates the residual of interest
            - group: str group from PINN to get derivatives and outputs from
            - input_vars: the 2 independent variables to plot against in the contour
            - time point: if time dependent, specify the time_point. The time interval and time col can be set when instanciating `Verification_module()`
            - eq_name: Name of equation, if None then use `equation.__name__` as the name of the equation
            - kwargs: Keyword arguments for `plot_2D()` see `Plotter()` object for more details
        '''
        with torch.no_grad():    
            device = PINN.net.parameters().__next__().device
            
            if time_point is not None:
                self.set_time_point(time_point)
            output = PINN(self.contour_points.to(device))['all']
            res = equation(**output).pow(2).cpu()
        
        if eq_name is None:
            eq_name = equation.__name__

        x,y = self.xs[input_vars[0]],self.xs[input_vars[1]]
        return self.plot_2D('contour',x,y,res,input_vars=input_vars,output_var=f"{eq_name} Residual",**kwargs)


    def log(self,epoch:int,**kwargs) -> None:
        '''
        log data into `self.logger` at some specified epoch. See `Tracker()` object for more details.
        '''
        self.logger.log(epoch,**kwargs)
    
    def log_to_DataFrame(self,fillna = None) -> pd.DataFrame:
        '''
        Convert the logger into a pandas dataframe for further processing.

        inputs:
            - fillna : Value to replace NaN type values in the dataframe

        output:
            - Dataframe Object of the logged data
        '''
        return self.logger.to_DataFrame(fillna)
    
    def log_to_csv(self,filename:str,fillna = None) -> None:
        '''
        save logged data into csv format by first converting into a pandas Dataframe:

        inputs:
        - fillna : Value to replace NaN type values in the dataframe

        '''
        self.log_to_DataFrame(fillna).to_csv(filename)


    def add_data(self,x_ref:torch.Tensor,y_ref:torch.Tensor,output_var:str):
        '''
        Add Ground truth data for comparison. Currently only supports 
        
        inputs:
            - x_ref : torch Tensor of size (N,D) that should be passable to a PINN network. first dim of x_ref must match first dim of y_ref
            - y_ref : torch Tensorr of size (N,) representing the data found a points in x_ref
            - output_var: str to name the ground truth data

        '''
        assert x_ref.shape[0] == y_ref.shape[0], 'First dim of x_ref and y_ref must match'
        self.ground_truth[output_var] = (x_ref,y_ref)
    

    def Lp_error(self,func:Union[nn.Module,Callable],output_var:str) -> torch.Tensor:
        '''
        Calculate the error between the network/function and the ground truth.

        Inputs:
            - func: `nn.Module()` or Callable function. if func is a callable function that is *not* a neural network (such as `DE_Getter()` calls), 
            then the function should be modified or created such that the output is a tensor of size (N,) where N is the size of the corresponding ground truth tensor
            - output_var: str output_var that refers to the ground truth data
        '''
        x,y = self.ground_truth[output_var]
        
        if isinstance(func,nn.Module): # If func is a network
            _,output_dict = self.format(x,func,to_numpy=False)
            return output_dict[output_var] - y.cpu()
        else: # Assume net is callable function instead that return a tensor of size (N,)
            return func(x) - y

    def Lp_error_norm(self,func:Union[nn.Module,Callable],output_var:str,power:float = 1)-> torch.Tensor:
        '''
        Calculate the Lp error norm between the network/function and the ground truth.

        Inputs:
            - func: `nn.Module()` or Callable function. if func is a callable function that is *not* a neural network (such as `DE_Getter()` calls), 
            then the function should be modified or created such that the output is a tensor of size (N,) where N is the size of the corresponding ground truth tensor
            - output_var: str output_var that refers to the ground truth data
            - power: float power to pass into `Tensor().norm(p=power)`
        '''
        return (self.Lp_error(func,output_var)).norm(power)
    
    def Lp_relative_error_norm(self,func:Union[nn.Module,Callable],output_var:str,power:float = 1)-> torch.Tensor:
        '''
        Calculate the relative Lp error norm between the network/function and the ground truth.

        Inputs:
            - func: `nn.Module()` or Callable function. if func is a callable function that is *not* a neural network (such as `DE_Getter()` calls), 
            then the function should be modified or created such that the output is a tensor of size (N,) where N is the size of the corresponding ground truth tensor
            - output_var: str output_var that refers to the ground truth data
            - power: float power to pass into `Tensor().norm(p=power)`
        '''
        _,y = self.ground_truth[output_var]
        error_norm = self.Lp_error_norm(func,output_var,power)
        return error_norm/y.norm(power)

    def plot_ref(self,input_vars:List[str],output_var:str,**kwargs):
        '''
        Plot reference solution
        '''
        xs,u_ref = self.ground_truth[output_var]
        
        x,y = xs[:,self.input_vars.index(input_vars[0])],xs[:,self.input_vars.index(input_vars[1])]

        return self.plot_2D('contour',x,y,u_ref,input_vars=input_vars,output_var=output_var,**kwargs)

    def Lp_error_plot(self,net:Union[nn.Module,Callable],input_vars:List[str],output_var:str,power:int = 1,**kwargs):
        '''
        Plot Lp error between ground truth and prediction
        '''
        error = torch.abs(self.Lp_error(net,output_var)).pow(power)
        xs,_ = self.ground_truth[output_var]
        x,y = xs[:,self.input_vars.index(input_vars[0])],xs[:,self.input_vars.index(input_vars[1])]

        return self.plot_2D('contour',x,y,error,input_vars=input_vars,output_var=output_var,**kwargs)


    def animate_plot(self,func):
        pass