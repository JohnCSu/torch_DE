import torch
import torch.nn as nn
from functorch import jacrev,jacfwd,vmap,make_functional
from .Engines import *
from typing import Union
from .utils import Data_handler
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
    def __init__(self,net:nn.Module,input_vars :list = None , output_vars: list= None,derivatives: list= None,deriv_method = 'AD',*args, **kwargs) -> None:
        '''
        Object to extract derivatives from a pytorch network via AD. Simplifies the process by abstracting away indexing to get specific derivatives with
        a dictionary with strings as keys.

        Inputs:
        input_vars: List | tuple of strings of what the input variable/independent variables should be called. Currently only single characters are supported
        
        output_vars:List | tuple of strings of what the output variable/dependent variables should be called.
        
        derivatives: List | tuple of strings of what derivatives to extract. The syntax is the dependent variable name followed by a number of independent variables.
        output and input variable are seperated by an underscore. For example, 'u_xx' will extract the second derivative of u with respect to x

        deriv_method: string | engine Object method to use to extract the derivatives from the neural network. use the following strings for pre implemented engines:
            AD: (default) obtain the derivatives using automatic differentiation/backprop.
            stein : Obtain gradients via stein's identity without backprop. Only works for first and second order derivatives
            engine Object: Pass in yuor own engine object to extract derivatives see engine for more details

        '''
        # super().__init__()
        self.net = net
        self.derivatives = {}
        if input_vars is not None and output_vars is not None:
            self.set_vars(input_vars,output_vars)
        if derivatives is not None:
            self.set_derivatives(derivatives)
            self.set_deriv_method(deriv_method)
        
        
        
    def set_vars(self,input_vars: iter,output_vars: iter,net_check = True):
        self.input_vars = input_vars
        self.output_vars = output_vars
        
        self.input_vars_idx = {input_var: i for i,input_var in enumerate(input_vars) }
        
        self.output_vars_idx ={output_var: i for i,output_var in enumerate(output_vars) }

        #Add the network evaluation output to this dictionary
        self.derivatives.update({output_var: (i,) for i,output_var in enumerate(output_vars) })

        if net_check is True:
            #Check that networks input and output match
            initial_device = next(self.net.parameters()).device
            test_net = self.net.cpu()
            try:
                x = torch.zeros((1,len(input_vars)))
                y = test_net(x)

                #Check output size matches number of output_vars given
                assert y.shape[1] == len(output_vars), f'The output of the network of size {y.shape[1]} does not match the number of output variables given {len(output_vars)}'
                self.net = self.net.to(device=initial_device)
            except RuntimeError:
                print(f'The number of input vars provided {len(input_vars)} does not match the input size of the network')
                self.net = self.net.to(device=initial_device)

            
    def set_derivatives(self,derivatives : list ) -> None:
        #For now assume single character variable names --> Will need to update the function

        #If '_' is used multiple times an error is raised. How to split longer names with '-' ? looks ugly though
        
        for deriv in derivatives:
            #Checking if variables in each derivative have been defined
            dep_var, indep_vars = deriv.split('_')
            assert dep_var in self.output_vars, f'Output Variable {dep_var} does not exist'
            
            for indep_var in (indep_vars):
                assert indep_var in self.input_vars, f"Variable {indep_var} is not an input Variable"
            
            #Work out the derivatives we need 
            self.get_deriv_index(dep_var,tuple(indep_vars))


    def get_deriv_index(self,dep_var:str,indep_vars: list)-> None: 
        # ignoring batch dimension
        # Input will be : ('u',['x','x'] )
        # indep_vars is treated as a list. For future so can handle longer string names
        # 0th dimension is dependent vars, 2nd dim is 1st order derivs, 3rd is 2nd ...
        
        for i in range(1,len(indep_vars)+1):
            var = ''.join(indep_vars[0:i])
            deriv = f'{dep_var}_{var}'
            

            if deriv not in self.derivatives.keys():
                index = (self.output_vars_idx[dep_var],) + tuple(self.input_vars_idx[indep_var] for indep_var in var)
                self.derivatives[deriv] = index


    def set_deriv_method(self,deriv_method,**kwargs):
        '''
        Set how to generate the derivatives for the PINN. Note that derivatives to extract must be supplied before calling the derivative method
        
        deriv_method: string | engine Object method to use to extract the derivatives from the neural network. use the following strings for pre implemented engines:
            AD: (default) obtain the derivatives using automatic differentiation/backprop.
            FD: Obtain the derivatives via finite difference. Currently only supports 2D and upto 2nd order non-mixed derivatives

            stein : Obtain gradients via stein's identity without backprop. Only works for first and second order derivatives
            
            engine Object: Pass in yuor own engine object to extract derivatives. Must be already initialised

            
        kwargs: any keywords to initialize the engine. net and derivatives are automatically passed in
        '''
        if not isinstance(self.derivatives,dict):
            raise ValueError(f'The derivatives to extract has not been set properly instead a type of {type(self.derivatives)} was found')
        if isinstance(deriv_method,str):
            if deriv_method  == 'AD':
                self.deriv_method = AD_engine(self.net,self.derivatives)
            elif deriv_method == 'FD':
                pass
        elif isinstance(deriv_method,engine):
            self.deriv_method = deriv_method
        else:
            raise TypeError('deriv_method should be an engine class or appropriate string')




    def calculate(self,x : Union[torch.tensor,dict,Data_handler], **kwargs) -> dict:
        '''
        Extract the desired differentials from the neural network using ADE and functorch. The Engine 

        Args:
        x Union[torch.tensor,dict,Data_handler].
            - torch.tensor:  
            - dict | Data_handler. If the data is stored with a dict like object such as Data_handler (has dict methods keys(), values() and items()) then groups and group_sizes can be left blank

        **kwargs: keyword arguments depending on the method to extract derivatives

        With Autodiff Engine built into Torch DE we have the following additional kwargs:

        groups list | iterable : an ordered list or tuple of group names \\
        group_sizes : a list or tuple containing the sizes of each group must be invoked if groups is not None
        
        groups and groups sizes is used, it should be a concatenated tensor of all groups of tensors in corresponding order. if groups is None, all output is placed is place into one group

        By default engines will concatenate the dictionary of different groups together to form one single batched tensor. Set cat = False

        
        '''
        return self.deriv_method.calculate(x,**kwargs)
       

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

        


if __name__ == '__main__':
    # Collocation Points (From 0 to 2pi)
    t_col = torch.rand((998,1))*2*torch.pi

    # Initial conditions u(0) = 0 , u_t(0) = 1
    t_data = torch.tensor([0]).unsqueeze(-1)

    t = torch.cat([t_data,t_col])

    print(t.shape)
    net = nn.Sequential(nn.Linear(1,200),nn.Tanh(),nn.Linear(200,1))
    # Spring Equation

    PINN = DE_Getter(net = net)
    PINN.set_vars(input_vars= ['t'], output_vars= ['u'])
    PINN.set_derivatives(derivatives=['u_t','u_tt'])

    optimizer = torch.optim.Adam(params = net.parameters(), lr = 1e-3)

    # For Loop
    for i in range(1):
        output = PINN.calculate(t)
        


        print(PINN.derivatives)
        out = output['all']
        #Spring Equation is u_tt + u = 0. Notice we can easily call derivatives and outputs by strings rather than having to do
        #indexing
        residual = (out['u_tt'] + out['u']).pow(2).mean()

        #Data Fitting. In this case we know that the first element is the point t=0
        data = out['u'][0].pow(2).mean() + (out['u_t'][0] - 1).pow(2).mean()


        loss = data + residual
        print(f'Epoch {i} Total Loss{float(loss)}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

