from torch_DE.equations.Get_Derivatives import get_derivatives
from typing import Union,List,Tuple,Dict,Callable

def get_NavierStokes(dims:int = 2,steady_state:bool = False,incompressible:bool = True,Re= None) -> Tuple[List[str],List[str],List[str],Dict[str,Callable]]:
    '''
    Returns the Reynold non-dimensional NavierStokes equations and other helpful things for dims upto 3

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
    if incompressible is False: 
        raise ValueError('Only Incompressible flow is currently supported')
    
    if  dims > 3 or dims < 1 or not isinstance(dims,int):
        raise ValueError('Dims can only be int type and between 1 and 3')


    # Constructing the function definition string
    indep_vars = ('x','y','z')
    vels = ('u','v','w')
    Re = f'Re={Re}' if Re is not None else 'Re'
    function_front = 'Navier_Stokes'
    pressure_grads = [f'p_{ind_var}' for ind_var,_ in zip(indep_vars,range(dims))]

    functions ={}
    for i in range(dims):

        function_name = f'{function_front}_{indep_vars[i]}'
        main_var = vels[i]
        
        vels_in_dim = vels[:dims]

        main_var_t = f'{main_var}_t' if not steady_state else ''
        add_term = '' if main_var_t == '' else '+'

        derivs_1 = [f'{main_var}_{ind_var}' for ind_var,_ in zip(indep_vars,range(dims))]
        derivs_2 = [f'{main_var}_{ind_var}{ind_var}' for ind_var,_ in zip(indep_vars,range(dims))]
        

        NS = f'{main_var_t}{add_term}{ "+".join([f"{vel}*{deriv}" for vel,deriv in zip(vels,derivs_1)]) } + {pressure_grads[i]}-1/Re*({"+".join(derivs_2)})' 
        
        main_var_t_arg = f",{main_var_t}," if main_var_t != '' else ','
        
        func_code = f'''def {function_name}({','.join(vels_in_dim)}{main_var_t_arg}{','.join(derivs_1)},{','.join(derivs_2)},{pressure_grads[i]},{Re},**kwargs):\n\treturn {NS}'''
        

        compiled_func = compile(func_code,'<string>','exec')
        exec(compiled_func,locals(),functions)
    
    diverg = [f'{vels[i]}_{indep_vars[i]}' for i in range(dims)]
    incompres = f'''def incompressible({','.join(diverg)},**kwargs):\n\t return {'+'.join(diverg)}'''
    incomp_compile = compile(incompres,'<string>','exec')
    exec(incomp_compile,locals(),functions)

    
    input_vars = indep_vars[0:dims] + ('t',) if not steady_state else indep_vars[0:dims]
    output_vars = vels[0:dims] + ('p',)

    derivatives = get_derivatives(input_vars,output_vars,*list(functions.values()),merge=True )
    return list(input_vars),list(output_vars),derivatives,functions


if __name__ == '__main__':
    
    a,b,c,d = get_NavierStokes(2,steady_state=True,Re=100)
    print('hi')