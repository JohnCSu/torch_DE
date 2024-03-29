import torch
from functorch import jacrev,jacfwd,vmap,make_functional
from .diff_engine import engine

class AD_engine(engine):
    def __init__(self,net,derivatives,**kwargs):
        super().__init__()
        self.net = net
        self.derivatives = derivatives
        
        self.highest_order = self.find_highest_order(derivatives)
        self.autodiff_deriv_func = self.compose_autodiff_deriv_func(net)
            

    def add_derivative(self,derivatives):
        self.derivatives = derivatives
        self.highest_order = self.find_highest_order(derivatives)
            

    def compose_autodiff_deriv_func(self,net):
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

    def calculate(self,x : torch.tensor, groups:list =None,group_sizes: list = None, **kwargs):
        out_tuple = vmap(self.autodiff_deriv_func)(x)
        #We get a nested tuple
        #Form is (nth derivative,(n-1,(n-2)...,(f(x))))
        #Need to unwrap into a single tuple

        #Example of u_xx and u_x call for x^2

        derivs = []
        #Denest the tuple (Should change to a generator function so looks nicer and avoids appending)
        for _ in range(self.highest_order):
            dy,y_tuple = out_tuple
            derivs.append(dy)
            out_tuple = y_tuple
        #Last y_tuple is the network evaluation
        derivs.append(y_tuple)
        
        #Output is a dictionary with keys being the group name. We always have the 'all' group. value of output[key] is another dictionary where
        # the key is the derivative string (e.g. u_xx) and the value is the values for that derivative
        output = {'all' : self.assign_derivs(derivs)}
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
        

    def assign_derivs(self,derivs):
        
        #Should I turn this into a one liner?
        output = {}
        
        for deriv_var,idx in self.derivatives.items(): 
            #Highest derivs are at 0 and last eval is at -1
            order = len(idx)-1
            j = self.highest_order-order

            # print(order,j,deriv_var)
            #Slice(None) python trick. Represents the ':' when indexing like A[:,1,2]
            index = (slice(None),) + idx

            output[deriv_var] = derivs[j][index] 
        return output