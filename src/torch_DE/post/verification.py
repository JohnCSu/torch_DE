import torch
from typing  import Dict,Tuple,List,Union
from matplotlib import pyplot as plt
from pandas import DataFrame
import pandas as pd
from torch_DE.post import Plotter,Tracker
from torch_DE.continuous import DE_Getter
from torch import nn
class Verification_module(Plotter):
    '''
    Module to track and log errors and images. Inherits from `Plotter()` class and uses the `Tracker()` class via `self.logger`.

    This class implements some additional methods to help with plotting. For more details on kwargs see `Plotter()` and `Tracker()` classes

    '''
    def __init__(self,input_vars,output_vars,*,domain = None,test_points = None,time_int = None,time_col = -1):
        super().__init__(input_vars,output_vars)

        # self.input_vars :Union[List,Tuple]   = input_vars
        # self.output_vars:Union[List,Tuple]  = output_vars
        self.ground_truth       :Dict[str,Tuple[torch.Tensor,torch.Tensor]] = {}

        self.logger = Tracker()
        
        #domain and test_points must be XOR in regards to being Nonetype
        assert domain is None and test_points is None, 'domain or test points can\'t both be Nonetype!'
        assert domain is not None and test_points is not None, 'only one of inputs domain and test_point can be non Nonetype'

        self.domain = domain
        if self.domain is not None:
            self.set_contour_points(self.domain,resolution=100,time_interval = time_int,time_col=time_col) 
        elif test_points is None:
            self.contour_points = test_points
        
        self.xs = {input_var: self.contour_points[:,i] for i,input_var in enumerate(input_vars)}


    def plot_residual(self,PINN,equation,group,input_vars,time_point:float = None,eq_name:str = None,**kwargs):
        with torch.no_grad():
            if time_point is not None:
                self.set_time_point(time_point)
            output = PINN(self.contour_points)[group]
            res = equation(**output).pow(2)
        
        if eq_name is None:
            eq_name = equation.__name__

        x,y = self.xs[input_vars[0]],self.xs[input_vars[1]]
        return self.plot_2D('contour',x,y,res,input_vars=input_vars,output_var=f"Residual {eq_name}",**kwargs)


    def log(self,epoch,**kwargs) -> None:
        self.logger.log(epoch,**kwargs)
    
    def log_to_DataFrame(self,fillna = None) -> pd.DataFrame:
        return self.logger.to_DataFrame(fillna)
    
    def log_to_csv(self,filename,fillna = None) -> None:
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
    

    def error(self,func,output_var) -> torch.Tensor:
        x,y = self.ground_truth[output_var]
        
        if isinstance(func,nn.Module): # If func is a network
            _,output_dict = self.format(x,func,to_numpy=False)
            return output_dict[output_var] - y
        else: # Assume net is callable function instead that return a tensor of size (N,)
            return func(x) - y

    def error_norm(self,func,output_var,power = 1)-> torch.Tensor:
        return (self.error(func,output_var)).norm(power)
    
    def relative_error_norm(self,func,output_var,power = 1)-> torch.Tensor:
        _,y = self.ground_truth[output_var]
        error_norm = self.error_norm(func,output_var,power)
        return error_norm/y.norm(power)

    def plot_ref(self,input_vars,output_var,**kwargs):
        '''
        Plot reference solution
        '''
        xs,u_ref = self.ground_truth[output_var]
        
        x,y = xs[:,self.input_vars.index(input_vars[0])],xs[:,self.input_vars.index(input_vars[1])]

        return self.plot_2D('contour',x,y,u_ref,input_vars=input_vars,output_var=output_var,**kwargs)

    def error_plot(self,net,input_vars,output_var,power = 1,**kwargs):
        '''
        Plot Absolute error between ground truth and prediction
        '''
        error = torch.abs(self.error(net,output_var)).pow(power)
        xs,_ = self.ground_truth[output_var]
        x,y = xs[:,self.input_vars.index(input_vars[0])],xs[:,self.input_vars.index(input_vars[1])]

        return self.plot_2D('contour',x,y,error,input_vars=input_vars,output_var=output_var,**kwargs)


    def animate_plot(self,func):
        pass