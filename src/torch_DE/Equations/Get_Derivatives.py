import inspect
def get_derivatives(input_vars,output_vars,*equations,merge = True) -> tuple:
    remove_list = set(['kwargs'] + list(input_vars) + list(output_vars))

    derivatives = []
    for equation in equations:
        var_names = set(inspect.signature(equation).parameters.keys())
        to_remove = set()
        for var in var_names:
            if var in remove_list:
                to_remove.add(var)
            #This means that there is no underscore
            elif str.split(var,'_')[0] == var:
                to_remove.add(var)
            else:
                
                output_var,in_vars = str.split(var,'_')

                for input_var in in_vars:
                    if input_var not in input_vars:
                        to_remove.add(var)
                        break
                
                if output_var not in output_vars:
                    to_remove.add(var)
        
        var_names = var_names.difference(to_remove)
        
        derivatives.append(var_names)

    if merge:
        derivs = set()
        for dydx in derivatives:
            derivs.update(dydx)
        return tuple(derivs)
    
    return tuple(derivatives)

