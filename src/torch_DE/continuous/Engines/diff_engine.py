import torch
from functorch import jacrev,jacfwd,vmap,make_functional

class engine():
    def __init__(self) -> None:
        self.derivatives = None
        self.net = None
    def __call__(self,*args,**kwargs):
        return self.calculate(*args,**kwargs)

    def calculate(self):
        pass
    
    def find_highest_order(self,derivatives):
        #Checking if variables in each derivative have been defined
        highest_order = 0
        print(derivatives.keys())
        for deriv in derivatives.keys():
            if deriv.find('_') != -1:
                _, indep_vars = deriv.split('_')
                #Order Function
                order = len(indep_vars)
                if order > highest_order:
                    highest_order = order  
        return highest_order
    

    def aux_function(self,aux_func,is_aux = True) -> object:
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
        
        
