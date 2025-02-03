from functools import wraps
from torch import Tensor
import inspect
# from torch_DE.utils.data import PINN_group
def DE_func(func):
    '''
    Decorator to turn your function into form suitable for troch DE. This is primarily for functions defined explicitly using their input varible names
    e.g. Poisson2D(u_xx,u_zz,x,y,**kwargs). Important note that **kwargs keyword variable MUST be defined otherwise an error is raised. The kwargs will absorb any
    unused variables (e.g. u_x , u_z for the above equation)

    Otherwise functions must be defined using dictionaries as inputs in the form f(x,y) where  x is the dictionary containing the corresponding input data 
    and y is a dictionary containing output and derivatives of the network
    '''
    sig = inspect.signature(func)
    assert inspect.Parameter.VAR_KEYWORD in [param.kind for param in sig.parameters.values()] , 'To use DE_func, you need to have the **kwargs declared in your function input e.g foo(x,y,**kwargs)'
    
    @wraps(func)
    def DE_func_wrapper(network_input,network_output:dict[str,Tensor]):
        return func(**network_input.batchables,**network_input.unbatchables,**network_output)
    DE_func_wrapper.is_decorated = True
    DE_func_wrapper.base_func = func
    return DE_func_wrapper
