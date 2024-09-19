from sympy import Symbol
from typing import Union,Any
import torch
import torch.nn as nn

def to_Symbol(var):
    if isinstance(var,str):
        return Symbol(var)
    elif isinstance(var,Symbol):
        return var
    else:
        raise ValueError(f'variable is not of type string or sympy.Symbol. Found {type(var)} instead')  

def to_String(var):
    if isinstance(var,Symbol):
        return var.name
    elif isinstance(var,str):
        return var
    else:
        raise ValueError(f'variable is not of type string or sympy.Symbol. Found {type(var)} instead')  


class Deriv(Symbol):
    '''
        Creates a derivative symbol. Inherits from Symbol class in Sympy. Functions basically the same as a `sympy.Symbol` just with some
        extra string inputs to standardise how derivatives are represented

        inputs:
            - Output_var: str name of output var/depedent variable/variable being differentiated e.g the `y` in dy/dx
            - Input var: str | list[str] name of input var/independent variable that the output var is being differentiated with respect to e.g. the `x` in dy/dx
                if input_var is a list then this defines a mixed derivative e.g. ['x','z'] would define dy/dxz. must be used with notation standard
                For consistentcy with mixed derivatives, input var is always stored as a tuple (even if single input variable)
            - order: int the derivative order. must be greater than or equal to 1. This argument is ignored if input var is a list
            - notation: str how the Symbol name is constructed. derivatives are represented as the subscript method e.g. `u_xx` or `v_y`
                - 'standard' will represent string whereby the independent variable is repeated n times equal to varibale order e.g. `u_xxxx`
                - 'compact' will use a number at the end to indicate the order in the format `{output_var}_{input_var}_{order}`  e.g. `u_x_4`
    '''
    def __new__(cls, output_var: str, input_var: str, order: int = 1, notation: str = 'standard', **kwargs):
        '''
        Create the symbol before initializing attributes in __init__.
        This is necessary because sympy.Symbol uses the __new__ method for object creation.
        '''

        output_var = to_String(output_var)
        input_var,order,input_vars_str,_ = Deriv._format_input_vars(input_var,notation,order)
        assert order >= 1 and isinstance(order, int), 'order must be an integer greater than or equal to 1'
 
        # Construct the symbol name based on notation
        if notation == 'standard':
            name = f'{output_var}_{input_vars_str}'
        elif notation == 'compact':
            #Only non-mixed derivatives available for compact
            name = f'{output_var}_{input_var}_{order}'
        else:
            raise ValueError('notation must be either standard or compact')

        # Call Symbol's __new__ method to create a new Symbol instance
        return super().__new__(cls, name, **kwargs)


    def __init__(self,output_var:str,input_var:str,order:int = 1,notation = 'standard',**kwargs) -> None:

        assert order >= 1 and isinstance(order,int),'order must be an integer greater than or equal to 1'
        
        self._output_var = output_var
        self._input_var,self._order,_,self._is_mixed = Deriv._format_input_vars(input_var,notation,order)
        self.notation = notation

    @property
    def output_var(self):
        return self._output_var
    
    @property
    def input_var(self):
        return self._input_var
    
    @property
    def order(self):
        return self._order

    @property
    def is_mixed(self):
        return self._is_mixed

    def list_input_vars(self):
        if self.is_mixed:
            return self.input_var
        else:
            return self.input_var*self.order
        

    @staticmethod
    def _format_input_vars(input_var,notation,order):
        is_mixed = False

        if isinstance(input_var,(list,tuple)):
            order = len(input_var)
            assert notation == 'standard', 'Mixed derivatives must use standard notation'
            input_vars_str = ''.join(input_var)
            is_mixed = True

        elif isinstance(input_var,str):
            # input_vars is for string formatting
            input_vars_str = input_var*order
            input_var = (input_var,)

        return input_var,order,input_vars_str,is_mixed
    
    def previous_Deriv(self):
        assert self.order > 1, 'Order must be greater than one to have a lower order derivative'
        if self.is_mixed:
            return Deriv(self.output_var,self.input_var[:self.order-1])
        else: 
            return Deriv(self.output_var,self.input_var[0],order = self.order-1,notation=self.notation)
    



def derivs(derivs):
    '''
    Create a list of derivatives from a list of tuple of the following format: `(output_var,input_var,order,notation)` from `Deriv` class.
    Note that order and notation are optional inputs
    '''
    return list([Deriv(*d) for d in derivs])


class Variable_dict(dict):
    '''
    Allows accessing of dictionary using sympy Symbols or strings. Assumes that Symbol().name maps to the equivalent string e.g `Symbol('x')` maps to the same value as 'x'
    '''
    def __init__(self,dictionary:dict = None):
        super().__init__()
        if isinstance(dictionary,dict):
            for key,val in dictionary.items():
                self.__setitem__(key,val)

    def __setitem__(self, key: str, value) -> None:
        if isinstance(key,Symbol):
            key = key.name
        assert isinstance(key,str), 'Variable dict only support Sympy.Symbol or string as keys' 
        return super().__setitem__(key, value)
    
    def __getitem__(self, key: str):
        if isinstance(key,Symbol):
            key = key.name
        assert isinstance(key,str),'Variable dict only supports Sympy.Symbol or string as keys'
        return super().__getitem__(key)

class Variable_list(list):
    '''
    Act exactly like a list but has a different __contains__ method
    '''
    def __contains__(self,item):
        assert isinstance(item,(str,Symbol))

        item = item.name if isinstance(item,Symbol) else item

        for var in self:
            if isinstance(var,Symbol):
                var = var.name
            
            if var == item:
                return True
            
        return False