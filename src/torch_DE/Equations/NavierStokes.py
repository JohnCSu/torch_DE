from torch_DE.equations.Get_Derivatives import get_derivatives
from typing import Union,List,Tuple,Dict,Callable
from torch_DE.equations.de_func import DE_func

from functools import partial
import inspect

def NavierStokes_x(u,v,w, u_x,u_y,u_z, u_t, u_xx,u_yy,u_zz,p_x,Re, **kwargs):
    NS_x =u_t + u*u_x + v*u_y + w*u_z + p_x - 1/Re*(u_xx + u_yy + u_zz)
    return NS_x


def NavierStokes_y(u,v,w, v_x,v_y,v_z, v_t, v_xx,v_yy,v_zz,p_y,Re, **kwargs):
    NS_y =v_t + u*v_x + v*v_y + w*v_z + p_y - 1/Re*(v_xx + v_yy + v_zz)
    return NS_y

def NavierStokes_z(u,v,w, w_x,w_y,w_z ,w_t, w_xx,w_yy,w_zz,p_z,Re, **kwargs):
    NS_z =w_t + u*w_x + v*w_y + w*w_z + p_z - 1/Re*(w_xx + w_yy + w_zz)
    return NS_z


def incompressible(u_x,v_y,w_z,**kwargs):
    return u_x + v_y + w_z

def get_NavierStokes(dims:int = 2,steady_state:bool = False,Re= None) -> Tuple[List[str],List[str],List[str],Dict[str,Callable]]:
    '''
    Returns the Reynold non-dimensional incompressible NavierStokes equations and other helpful things for dims upto 3

    Inputs:
        dims            : int number of dimensions from 1 to 3
        steady_state    : bool (default False) of whether to calculate Navierstokes in steady state or transient 
        incompressible  : bool (default True) of if to have incompressible flow
        Re              : float | Nonetype whether to preinitialise the Reynolds number. Default None
    Returns:
        tuple: (input_vars,output_vars,function_dict)
            input_vars  : tuple of independent variables in the order of (x,y,z) depending on the number of dims, if transient then the variable t is appended to the end e.g. (u,v,t)
            output_vars : tuple of dependent variables in the order of (u,v,w) depending on the number of dims with pressure add to the end e,g (u,v,p)
            derivatives : tuple containing all the derivatives needed across all equations
            functions   : Dict containing the Navier stokes equations in `dims` dimensions and the continuity equation (incompressibility)
    '''
    t = tuple(['t'])
    if steady_state:
        t = tuple()
        steady_state = {
            'u_t':0,
            'v_t':0,
            'w_t':0
        }
    else:
        steady_state = {}
        # NS_x,NS_y,NS_z = (partial(NavierStokes_x,u_t = 0,v_t = 0,w_t =0,Re = Re), partial(NavierStokes_y,u_t = 0,v_t = 0,w_t =0,Re = Re), partial(NavierStokes_z,u_t = 0,v_t = 0,w_t =0,Re = Re))
    
    
    NS_x,NS_y,NS_z = (partial(NavierStokes_x,**steady_state,Re = Re), partial(NavierStokes_y,**steady_state,Re = Re), partial(NavierStokes_z,**steady_state,Re = Re))
    
    input_vars = ('x','y','z')
    output_vars = ('u','v','w')

    input_vars = input_vars[0:dims] + t
    output_vars = output_vars[0:dims] + ('p',)

    # variables = set( [inspect.signature(NS).parameters.keys() for NS in equations]  )

    
    if dims > 3 or dims < 1:
        raise ValueError('Dims can only be int type and between 1 and 3')
    elif dims == 2:
        NS_x,NS_y = partial(NS_x,w=0,u_z =0,u_zz = 0),partial(NS_y,w=0,v_z =0,v_zz = 0)
        incomp = partial(incompressible,w_z = 0)
    elif dims == 1:
        NS_x = partial(NS_x,v =0, w=0,u_z =0,u_y =0,u_yy = 0,u_zz = 0)
        incomp = partial(incompressible,v_y = 0,w_z = 0)
    
    names = ('NavierStokes_x','NavierStokes_y','NavierStokes_z')[0:dims] + ('incompressible',)
    equations = (NS_x,NS_y,NS_z)[0:dims] + (incomp,)
        
    derivatives = get_derivatives(input_vars,output_vars,*equations)
    equations = {name:DE_func(equation) for name,equation in zip(names,equations)}

    return input_vars,output_vars,derivatives,equations


if __name__ == '__main__':
    
    a,b,c,d = get_NavierStokes(2,steady_state=True,Re=100)
    print(a,b,c,d)