import torch
from typing  import Dict,Tuple,List,Union
from matplotlib import pyplot as plt

class Validation_module():
    def __init__(self,input_vars,output_vars):
        self.input_vars :Union[List,Tuple]   = input_vars
        self.output_vars:Union[List,Tuple]  = output_vars
        self.data       :Dict[str,Tuple[torch.Tensor,torch.Tensor]] = {}
    def format(self,x,net,to_numpy = True):
        if isinstance(net,torch.nn.Module):
            device = net.parameters().__next__().device
        else:
            device = 'cpu'

        with torch.no_grad():
            out = net(x.to(device))       
            out = out.cpu()
            x = x.cpu()
        
        if to_numpy:
            x = x.numpy()
            out = out.numpy() 

        input_dict = {input_var:x[:,i] for i,input_var in enumerate(self.input_vars)}
        output_dict = {output_var:out[:,i] for i,output_var in enumerate(self.output_vars)}

        return input_dict,output_dict
    def add_data(self,x_ref:torch.Tensor,y_ref:torch.Tensor,output_var):
        self.data[output_var] = (x_ref,y_ref)
    

    def error(self,net,output_var) -> torch.Tensor:
        x,y = self.data[output_var]
        _,output_dict = self.format(x,net,to_numpy=False)
        return output_dict[output_var] - y

    def error_norm(self,net,output_var,power = 1)-> torch.Tensor:
        return (self.error(net,output_var)).norm(power)
    
    def relative_error_norm(self,net,output_var,power = 1)-> torch.Tensor:
        _,y = self.data[output_var]
        error_norm = self.error_norm(net,output_var,power)
        return error_norm/y.norm(power)

    def plot_ref(self,input_vars,output_var,*,aspect_ratio = 'auto',show = True,save_name = None,levels = 100):
        
        xs,u_ref = self.data[output_var]
        
        x,y = x,y = xs[:,self.input_vars.index(input_vars[0])],xs[:,self.input_vars.index(input_vars[1])]
    
        fig,ax = plt.subplots()
        contour = ax.tricontourf(x,y,u_ref,levels = levels,cmap = 'jet')
        ax.set_title(f'Reference Solution For {output_var}')


        fig.colorbar(contour)
        ax.set_xlabel(input_vars[0])
        ax.set_ylabel(input_vars[1])
        ax.set_aspect(aspect_ratio)

        if show:
            fig.show()
        if save_name is not None:
            fig.savefig(save_name)
        plt.close(fig)

    def error_plot(self,net,input_vars,output_var,power = 1,epoch = None,aspect_ratio = 'auto',show = True,save_name = None,levels = 100):
        error = torch.abs(self.error(net,output_var)).pow(power)
        xs,_ = self.data[output_var]
        x,y = xs[:,self.input_vars.index(input_vars[0])],xs[:,self.input_vars.index(input_vars[1])]
    
        fig,ax = plt.subplots()
        contour = ax.tricontourf(x,y,error,levels = levels,cmap = 'jet')
        epoch_str = f'at epoch {epoch}' if epoch is not None else ''
        ax.set_title(f'L{power} Error For {output_var} {epoch_str}')
        
        fig.colorbar(contour)
        ax.set_xlabel(input_vars[0])
        ax.set_ylabel(input_vars[1])
        ax.set_aspect(aspect_ratio)

        if show:
            fig.show()
        if save_name is not None:
            fig.savefig(save_name)
        
        plt.close(fig)


