import torch
import torch.nn as nn

def get_activation_function(activation):
    if isinstance(activation,str):
        activation = getattr(torch,activation)
    return activation


class DE_Module(nn.Module):
    '''
    Basically a regular nn.Module but has some extra functions
    '''
    def __init__(self) -> None:
        super().__init__()
    
    def set_activation_function(self,activation_function,adaptive = False):
        activation_function = get_activation_function(activation_function)
        if adaptive:
            return adaptive_function(activation_function)
        else:
            return activation_function


class adaptive_function(nn.Module):
    def __init__(self,activation_function) -> None:
        super().__init__()
        self.activation_function = activation_function
        self.weight = nn.Parameter(data = torch.ones(1))
    def forward(self,x):
        return self.activation_function(self.weight*x)

class MLP(nn.Module):
    '''
    Simple Multi Layer Perceptron Network. Good as a baseline.

    in_features: size of model input
    out_features: size of model output
    hidden_features: number of features in each hidden layer
    num_hidden_layers: number of hidden layers of network
    activation: activation function of model. type string will use the correspong torch function or you can pass your own activation function.

    '''
    def __init__(self,in_features : int,out_features: int,hidden_features: int,num_hidden_layers: int,activation = 'tanh') -> None:
        super().__init__()
        self.linear_in = nn.Linear(in_features,hidden_features)
        self.linear_out = nn.Linear(hidden_features,out_features)
        
        
        self.activation = get_activation_function(activation)
        self.layers = nn.ModuleList([self.linear_in] + [nn.Linear(hidden_features, hidden_features) for _ in range(num_hidden_layers)  ])
        
         
    def forward(self,x):
        for layer in self.layers:
            x = self.activation(layer(x))
    
        return self.linear_out(x)

class Fourier_Net(DE_Module):
    def __init__(self,in_features : int,out_features: int,hidden_features: int,num_hidden_layers: int,num_freq,activation = 'tanh',adaptive_activation = False) -> None:
        super().__init__()
        self.f = nn.Linear(in_features,num_freq)
        self.linear_out = nn.Linear(hidden_features,out_features)
        self.activation = self.set_activation_function(activation,adaptive_activation)

        self.layers = nn.ModuleList([nn.Linear(num_freq*2,hidden_features)] + [nn.Linear(hidden_features, hidden_features) for _ in range(num_hidden_layers-1)])
    def forward(self,x):
        encode = torch.cat([torch.sin(2*torch.pi*self.f(x)),torch.cos(2*torch.pi*self.f(x))],dim = -1)
        for layer in self.layers:
            encode = self.activation(layer(encode))
        return self.linear_out(encode)


class Modified_Fourier_Net(DE_Module):
    '''
    Network described in Nvidia Modulus:
    https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/theory/architectures.html#modified-fourier-network 
    
    A variation of the network proposed by Wang et al: https://arxiv.org/abs/2001.04536 

    '''
    def __init__(self,in_features : int,out_features: int,hidden_features: int,num_hidden_layers: int,num_freq,activation = 'sigmoid',adaptive_activation = False) -> None:
        super().__init__()
        self.f = nn.Linear(in_features,num_freq)
        self.transform_layers = nn.ModuleList([nn.Linear(num_freq*2,hidden_features) for _ in range(2)])
        self.H1 = nn.Linear(in_features,hidden_features)
        self.linear_out = nn.Linear(hidden_features,out_features)
        self.activation = self.set_activation_function(activation,adaptive_activation)

        self.layers = nn.ModuleList([nn.Linear(hidden_features, hidden_features) for _ in range(num_hidden_layers-1)])

    def forward(self,x):
        encode = torch.cat([torch.sin(2*torch.pi*self.f(x)),torch.cos(2*torch.pi*self.f(x))],dim = -1)
        T1,T2 =[ self.activation(trans(encode)) for trans in self.transform_layers]
        H = self.activation(self.H1(x))

        for layer in self.layers:
            Z = self.activation(layer(H))
            H = (1-Z)*T1 + Z*T2
        return self.linear_out(H)
    

def Siren_Weight_init(layer:nn.Module):
    if isinstance(layer,nn.Linear):
        n = layer.in_features
        bound = torch.sqrt(6/n)
        torch.nn.init.uniform_(layer.weight,-bound,bound)
        
class Siren_Net(DE_Module):
    def __init__(self,in_features : int,out_features: int,hidden_features: int,num_hidden_layers,w0 :float = 30.,adaptive_activation = False):
        super().__init__()
        self.w0 = w0
        self.linear_in = nn.Linear(in_features,hidden_features)
        
        self.linear_out = nn.Linear(hidden_features,out_features)
        self.activation =  self.set_activation_function('sin',adaptive_activation)
        self.layers = nn.ModuleList( [nn.Linear(hidden_features, hidden_features) for _ in range(num_hidden_layers)  ])

        #Apply weighting scheme
        self.apply(Siren_Weight_init)
        #Apply w0 to initial weight matrix
        bound = self.w0/in_features
        self.linear_in.weight.uniform_(-bound,bound)
         
    def forward(self,x):
        x = self.linear_in(x)
        for layer in self.layers:
            x = self.activation(layer(x))
    
        return self.linear_out(x) 



if __name__ == '__main__':
    net = Fourier_Net(3,3,10,5,5)
    net = Modified_Fourier_Net(3,3,10,5,5)
    x = torch.rand([1,3])

    net(x)


