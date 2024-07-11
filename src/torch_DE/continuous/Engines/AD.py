import torch
from torch.func import jacrev,jacfwd,vmap
from torch_DE.continuous.Engines import engine
from typing import Union,Dict,List,Callable,Iterable
from torch_DE.utils.data import PINN_dict
class AD_engine(engine):
    def __init__(self,net,derivatives,**kwargs):
        super().__init__()
        self.net = net
        self.derivatives = derivatives
        self.output_vars = self.get_output_vars(derivatives)
        self.highest_order = self.find_highest_order(derivatives)
        self.autodiff_deriv_func = self.compose_autodiff_deriv_func(net)
            

    def add_derivative(self,derivatives):
        self.derivatives = derivatives
        self.highest_order = self.find_highest_order(derivatives)
            

    def compose_autodiff_deriv_func(self,net:torch.nn.Module) -> Callable:
        '''
        Creates the function that when a tensor is passed into the function, it returns all the gradients up to the highest order. Note that this return ALL
        gradients (i.e the jacobian or Hessian) so another function is needed to pick out the gradients you really want
        '''
        deriv_function = []
        for _ in range(1,self.highest_order+1):
            # if (i % 2) == 1: # if odd use jacrev 
            #     deriv_function.append(jacrev)
            # else:
            #     deriv_function.append(jacfwd)
            deriv_function.append(jacrev)

        # self.derivative_function,self.params = make_functional(self.net)
        derivative_function = net
        is_aux = False
        for jac_func in deriv_function:
            derivative_function = jac_func(self.aux_function(derivative_function,is_aux),has_aux = True)
            is_aux = True
        return derivative_function

    def calculate(self,x : Union[torch.Tensor,dict,PINN_dict], target_groups:Union[str,List,tuple] = None, **kwargs) -> Dict[str, Dict[str,torch.Tensor]]:
        '''
        Calculate derivatives using autodiff via functorch

        Input:
            x: Union[torch.Tensor,dict,Data_handler]: either tensor or a dictionary of tensors represent input to the network 
            target group: str (default None) The group that will be differentiated via autodiff. if None all inputs are differentiated

        Returns
            Output_dict: Dict
        '''
        if isinstance(x,PINN_dict):
            if target_groups is not None:
                target_groups = [target_groups] if isinstance(target_groups,str) else target_groups 
                x_d,groups,group_sizes = self.cat_groups({target_group:x[target_group] for target_group in target_groups })
                derivs = self.autodiff(x_d)
                output_derivs = self.group_output(derivs,groups,group_sizes)
                output_dict = self.net_pass_from_dict(x,exclude = target_groups )
                output_dict.update(output_derivs)

            else:
                x,groups,group_sizes = self.cat_groups(x)
                output_dict = self.group_output(self.autodiff(x),groups,group_sizes)
        elif isinstance(x,torch.Tensor):
            output_dict = self.group_output(self.autodiff(x))

        return output_dict
        
    def autodiff(self,x:torch.Tensor) -> List[torch.Tensor]:
        '''
        When using functorch, the output is wrapped in a nested tuples of size 2 e.g Form is (nth derivative,(n-1,(n-2)...,(f(x))))
        and is reveresed so the network evaluation ("0th derivative") is the last element (deepest tuple)

        Goal of function is to unwrap this tuple and then reverse the order so the jth element corresponds to the jth derivative
        '''
        out_tuple = vmap(self.autodiff_deriv_func)(x)
        #We get a nested tuple
        #Form is (nth derivative,(n-1,(n-2)...,(f(x))))
        #Need to unwrap into a single tuple and reverse order

        derivs = []
        #Denest the tuple (Should change to a generator function so looks nicer and avoids appending)
        for _ in range(self.highest_order):
            dy,y_tuple = out_tuple
            derivs.append(dy)
            out_tuple = y_tuple
        #Last y_tuple is the network evaluation so we need to reverse the order so the ith element in the tuple is the ith derivative
        derivs.append(y_tuple)
        return derivs[::-1]
        
    def group_output(self,derivs:Dict[str,torch.Tensor],groups:str=None,group_sizes:List =None,target_group:str = None) -> Dict[str,Dict[str,torch.Tensor]]:
        '''
        Put the output data into a nicely formatted dictionary so we don't have to use indexing
        '''
        #Output is a dictionary with keys being the group name. We always have the 'all' group. value of output[key] is another dictionary where
        # the key is the derivative string (e.g. u_xx) and the value is the values for that derivative
        output = {'all' if target_group is None else target_group : self.assign_derivs(derivs)}
        #From Group size determine start of batching

        if groups is None:
            return output

        #If we have groups, we need to partition the batch 
        idx_start = 0
        for group,g1 in zip(groups,group_sizes):
            idx_end = idx_start + g1
            group_deriv = [deriv[idx_start:idx_end] for deriv in derivs]
            output[group] = self.assign_derivs(group_deriv)
            idx_start = idx_end
        
        return output
        

    def assign_derivs(self,derivs:Dict[str,torch.Tensor]) -> Dict[str,torch.Tensor] :
        '''
        Sort out the derivative output to place in dictionary form

        derivs: list of tensors where each tensor represents the output/derivative of the PINN. The jth element represents the jth derivative. 
            the 0th element represents the network evaluation u, 1st is u_x ... etc  
        '''

        #Should I turn this into a one liner?
        output = {}
        for deriv_var,idx in self.derivatives.items(): 
            #The jth element represents the jth order derivative
            j = len(idx) - 1
            #Slice(None) python trick. Represents the ':' when indexing like A[:,1,2]
            index = (slice(None),) + idx
            output[deriv_var] = derivs[j][index] 
        return output