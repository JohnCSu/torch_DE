import torch
import torch.nn as nn
from functorch import jacrev,jacfwd,vmap,make_functional



def aux_function(aux_func,is_aux = True) -> object:
    #aux_func has the output of (df,f) so we need it to output (df,(df,f))
    
    def initial_aux_func(x:torch.tensor) -> tuple[torch.tensor,torch.tensor]:
        out = aux_func(x)
        return (out,out)
    
    def inner_aux_func(x:torch.tensor) -> tuple[torch.tensor,torch.tensor]:
        out = aux_func(x)
        return (out[0],out)
    
    if is_aux:
        return inner_aux_func
    else:
        return initial_aux_func
    
    


class DE_Getter():
    def __init__(self,net,input_vars :list =None, output_vars = None,derivatives = None) -> None:
        '''
        Object to extract derivatives from a pytorch network via AD. Simplifies the process by abstracting away indexing to get specific derivatives with
        a dictionary with strings as keys.

        Inputs:
        input_vars: List | tuple of strings of what the input variable/independent variables should be called. Currently only single characters are supported
        output_vars:List | tuple of strings of what the output variable/dependent variables should be called.
        
        derivatives: List | tuple of strings of what derivatives to extract. The syntax is the dependent variable name followed by a number of independent variables.
        output and input variable are seperated by an underscore. For example, 'u_xx' will extract the second derivative of u with respect to x

        '''
        # super().__init__()
        self.net = net
        self.derivatives_index = {}
        self.set_vars(input_vars,output_vars)
        if derivatives is not None:
            self.set_derivatives(derivatives = derivatives)

    def set_vars(self,input_vars: iter,output_vars: iter):
        self.input_vars = input_vars
        self.output_vars = output_vars
        if input_vars is not None:
            self.input_vars_idx = {input_var: i for i,input_var in enumerate(input_vars) }
        if output_vars is not None:
            self.output_vars_idx ={output_var: i for i,output_var in enumerate(output_vars) }

            #Add the network evaluation output to this dictionary
            self.derivatives_index.update({output_var: (i,) for i,output_var in enumerate(output_vars) })


    def set_derivatives(self,derivatives : list ) -> None:
        #For now assume single character variable names --> Will need to update the function

        #If '_' is used multiple times an error is raised. How to split longer names with '-' ? looks ugly though
        
        self.highest_order = 0
        for deriv in derivatives:
            #e.g. u_xx

            #Checking Function
            dep_var, indep_vars = deriv.split('_')
            assert dep_var in self.output_vars, f'Output Variable {dep_var} does not exist'
            
            for indep_var in (indep_vars):
                assert indep_var in self.input_vars, f"Variable {indep_var} is not an input Variable"
            
            #Order Function
            order = len(indep_vars)
            if order > self.highest_order:
                self.highest_order = order  
            #Work out the derivatives we need 
            self.get_deriv_index(dep_var,tuple(indep_vars))

        #Create dict with same keys as derivative index
        self.output = {key:None for key in self.derivatives_index.keys()}
        self.compose_derivative()

    def get_deriv_index(self,dep_var:str,indep_vars: list)-> None: 
        # ignoring batch dimension
        # Input will be : ('u',['x','x'] )
        # indep_vars is treated as a list. For future so can handle longer string names
        # 0th dimension is dependent vars, 2nd dim is 1st order derivs, 3rd is 2nd ...
        
        for i in range(1,len(indep_vars)+1):
            var = ''.join(indep_vars[0:i])
            deriv = f'{dep_var}_{var}'
            

            if deriv not in self.derivatives_index.keys():
                index = (self.output_vars_idx[dep_var],) + tuple(self.input_vars_idx[indep_var] for indep_var in var)
                self.derivatives_index[deriv] = index


    def compose_derivative(self) -> dict:
        deriv_function = []
        for i in range(1,self.highest_order+1):
            # if (i % 2) == 1: # if odd use jacrev 
            #     deriv_function.append(jacrev)
            # else:
            #     deriv_function.append(jacfwd)
            deriv_function.append(jacrev)

        # self.derivative_function,self.params = make_functional(self.net)
        self.derivative_function = self.net
        is_aux = False
        for jac_func in deriv_function:
            self.derivative_function = jac_func(aux_function(self.derivative_function,is_aux),has_aux = True)
            is_aux = True



    def calculate(self,x : torch.tensor, groups:list =None,group_sizes: list = None, **kwargs) -> dict:
        '''
        Extract the desired differentials from the neural network using ADE and functorch

        Args:
        x (torch.tensor) : input tensor. If groups and groups sizes is used, it should be a concatenated tensor of all groups of tensors in corresponding order \n
        groups (list | tuple) : a list of names for each group of output. They should be ordered the same way x was concatenated. If none, then input tensor is treated as a single group. Must be used in conjunction with group_sizes \n
        groups__sizes (list | tuple): a list of ints describing the different lengths of each group in the input tensor x. Must be used in conjunction with groups \n

        Returns:\n 
        if groups == None:
            output (dict) : a dictionary of the form output[derivative_name] = tensor

        if groups != None:
            output (dict) : a dictionary containing another dictionary of the form output[group_name][derivative_name] = tensor
        
        To Do:
        Should the option of output[derivative_name][group_name] be considered?
        - group_name dependent derivatives (e.g. for a no slip wall we would only care about the velocity u and v)
        - Networks that have multiple inputs that maynot be tensors e.g. latent variables
            
        '''

        out = vmap(self.derivative_function)(x)
        #We get a nested tuple
        #Form is (nth derivative,(n-1,(n-2)...,(f(x))))
        #Need to unwrap into a single tuple

        #Example of u_xx and u_x call for x^2
        out_tuple = out 
    
        derivs = []
        #Denest the tuple (Should change to a generator function so looks nicer and avoids appending)
        for _ in range(self.highest_order):
            dy,y_tuple = out_tuple
            derivs.append(dy)
            out_tuple = y_tuple
        #Last y_tuple is the network evaluation
        derivs.append(y_tuple)
        

        # if groups is None:
        #     return self.assign_derivs(derivs)
        # else:
        output = {'all' : self.assign_derivs(derivs)}
        #From Group size determine start of batching

        if groups is None:
            return output

        idx_start = 0
        for group,g1 in zip(groups,group_sizes):
            idx_end = idx_start + g1
            group_deriv = [deriv[idx_start:idx_end] for deriv in derivs]
            output[group] = self.assign_derivs(group_deriv)
            idx_start = g1
        
            
        return output
        

    def assign_derivs(self,derivs):
        
        #Should I turn this into a one liner?
        output = {}
        
        for deriv_var,idx in self.derivatives_index.items(): 
            #Highest derivs are at 0 and last eval is at -1
            order = len(idx)-1
            j = self.highest_order-order

            # print(order,j,deriv_var)
            #Slice(None) python trick. Represents the ':' when indexing like A[:,1,2]
            index = (slice(None),) + idx

            output[deriv_var] = derivs[j][index] 
        return output

    def __call__(self, *args, **kwds) -> dict:
        '''
        Extract the desired differentials from the neural network using ADE and functorch
            Calls calculate method

        Args:
        x (torch.tensor) : input tensor. If groups and groups sizes is used, it should be a concatenated tensor of all groups of tensors in corresponding order \n
        groups (list | tuple) : a list of names for each group of output. They should be ordered the same way x was concatenated. If none, then input tensor is treated as a single group. Must be used in conjunction with group_sizes \n
        groups__sizes (list | tuple): a list of ints describing the different lengths of each group in the input tensor x. Must be used in conjunction with groups \n
        
        Returns:\n 
        if groups == None:
            output (dict) : a dictionary of the form output[derivative_name] = tensor

        if groups != None:
            output (dict) : a dictionary containing another dictionary of the form output[group_name][derivative_name] = tensor
        
        To Do:
        Should the option of output[derivative_name][group_name] be considered?
        - group_name dependent derivatives (e.g. for a no slip wall we would only care about the velocity u and v)
        - Networks that have multiple inputs that maynot be tensors e.g. latent variables
            
        '''
        return self.calculate(*args,**kwds)

    def test_setup(self,input_tensor = None):
        #If no input give, assume tensor size of shape (1,len(input_vars))
        if input_tensor is None:
            input_tensor = torch.zeros((1,len(self.input_vars)))


        #Test 1, check input size is correct and output size is correct
        try:
            y = a(x)
        except RuntimeError as existing_error:
            error = 'Theres likely a mismatch with the input tensor specified and the shape the network was expecting'
            error_text = f'{str(existing_error)}\n{error}' 
            
            raise RuntimeError(error_text)
            

        #Check that the output matches 
        # Ignore Batch size
        assert y.shape[1:] == len(self.output_vars)

        # Test 2, We can actually extract the derivatives

        

