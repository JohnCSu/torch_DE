import torch
from typing import Callable,Union,Dict,Iterable
import warnings


def set_time(time_type:str,*tensors:torch.Tensor,time_interval:Union[list,tuple] = None,point:float = None,col = -1,inplace = False):
    '''
    Set the time col of a tensor or a list of tensors. This is used if the time column already exists. It is assumed of the shape (B,M). If you want to add a time column
    see `add_time()`. Here one can specify the specific col that pertains to the time variable (e.g. if it is the first column or inbetween)

    input args:
        - time_type: str 
            - `random interval` randomly sample time points from variable `time_interval` of [a,b]
            - `random point` randomly sample a single time point inside the interval [a,b]. i.e. all points in a tensor will have the same time point
            - `single point` set a specific time point for all points in a tensor given by keyword arguement `point`
        - *tensors: iterable of args of tensors to add time component 
        - col: int  = -1 the column relating the time column to.
        - point: float the time point to set the tensors with
        - time_interval: list | tuple: time interval (a,b) to sample from. Must not be None if used with `random interval` or `random point`
        - inplace: bool Whether to perform the operation in place or create a new tensor. Default False
        
    Output:
        - a: list | tensor. If a single tensor is provided, return a tensors with time col added to it. If multiple tensors are provided, return a list of tensors
            if inplace then return None
    '''

    a = []
    for tensor in tensors:
        if inplace:
            t = tensor
        else:
            t = tensor.clone()
        
        assert len(t.shape) == 2
        if time_type == 'random interval':
            assert time_interval is not None, 'variable time_interval must not be None if time_type = "random interval"'
            add_random_time(t,time_interval,col)
        
        elif time_type == 'random point':
            assert time_interval is not None, 'variable time_interval must not be None if time_type = "random point"'
            add_random_time_point(t,time_interval,col)
        
        elif time_type == 'single point':
            assert point is not None, 'variable point must not be None if time_type = "single point"'
            add_time_point(t,point,col)
        else:
            raise ValueError(f'time type is a string of either "random interval","random point" or "single point" Got {time_type} string instead')
        a.append(t)
    
    if inplace:
        return None
    
    if len(a) == 1:
        return a[0]
    else:
        return a


def add_time(time_type:str,*tensors,time_interval:Union[list,tuple] = None,point:float = None,dim:int = -1):
    '''
    Add a time col to a tensor or a list of tensors

    input args:
        - time_type: str 
            - `random interval` randomly sample time points from variable `time_interval` of [a,b]
            - `random point` randomly sample a single time point inside the interval [a,b]. i.e. all points in a tensor will have the same time point
            - `single point` set a specific time point for all points in a tensor given by keyword arguement `point`
        - *tensors: iterable of args of tensors to add time component 
        - dim: int  = -1 the dimension to add the time column to. The time column is always set as the last column
        - point: float the time point to set the tensors with
        - time_interval: list | tuple: time interval (a,b) to sample from. Must not be None if used with `random interval` or `random point`
        

    Output:
        - a: list | tensor. If a single tensor is provided, return a tensors with time col added to it. If multiple tensors are provided, return a list of tensors
    '''

    if bool(point) ^ bool(time_interval) is False:
        warnings.warn('point and time interval are mutually exlusive keyword arguements') 

    a = []
    for tensor in tensors:
        t = add_time_col(tensor,dim)
        assert len(t.shape) == 2
        if time_type == 'random interval':
            assert time_interval is not None, 'variable time_interval must not be None if time_type = "random interval"'
            add_random_time(t,time_interval,-1)
        
        elif time_type == 'random point':
            assert time_interval is not None, 'variable time_interval must not be None if time_type = "random point"'
            add_random_time_point(t,time_interval,-1)
        
        elif time_type == 'single point':
            assert point is not None, 'variable point must not be None if time_type = "single point"'
            add_time_point(t,point,-1)
        else:
            raise ValueError(f'time type is a string of either "random interval","random point" or "single point" Got {time_type} string instead')
        a.append(t)
    
    if len(a) == 1:
        return a[0]
    else:
        return a
def add_time_col(tensor,dim):
    return torch.cat([tensor,torch.zeros((tensor.shape[0],1),device=tensor.device)],dim = dim)

def add_time_point(tensor,t_point,axis = -1):
    tensor[:,axis] = t_point*torch.ones((tensor.shape[0]),device=tensor.device)


def add_random_time_point(tensor,interval,col = -1):
    a,b = interval 
    tensor[:,col] = torch.ones((tensor.shape[0]),device=tensor.device)*(torch.rand(1,device=tensor.device)*(b-a)+a)

def add_random_time(tensor,interval,col = -1):
    a,b = interval
    tensor[:,col] = torch.rand((tensor.shape[0]),device=tensor.device)*((b-a)+ a)
