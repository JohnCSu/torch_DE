from matplotlib import pyplot as plt
import torch
from typing import Union,Iterable

from shapely import points,box
import geopandas as gpd

class Plotter():
    def __init__(self,input_vars:Union[list,tuple],output_vars:Union[list,tuple]) -> None:
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.contour_points = None
        self.domain =None
    def format(self,x,net):
        if isinstance(net,torch.nn.Module):
            device = net.parameters().__next__().device
        else:
            device = 'cpu'

        with torch.no_grad():
            out = net(x.to(device))       
            out = out.cpu().numpy()
            x = x.cpu().numpy()
        
        input_dict = {input_var:x[:,i] for i,input_var in enumerate(self.input_vars)}
        output_dict = {output_var:out[:,i] for i,output_var in enumerate(self.output_vars)}

        return input_dict,output_dict
    
    def plot(self,x:torch.Tensor,net:torch.nn.Module,input_var,output_var,*,save_dir = None,epoch = None,show = True,**kwargs):
        input_dict,output_dict = self.format(x,net)
        title = f'{output_var} vs {input_var} {f"at epoch {epoch}" if epoch is not None else ""}'
        plt.plot(input_dict[input_var],output_dict[output_var],**kwargs)
        plt.title(self.title(title,output_var,input_var,epoch))
        plt.xlabel(input_var)
        plt.ylabel(output_var)


        if save_dir is not None:
            plt.savefig(f'Plot_{output_var}_vs_{input_var}{f"_{epoch}" if epoch is not None else ""}.jpg')

        if show:
            plt.show()

    
    @staticmethod
    def _contour_plot(x,y,z,**kwargs):

        if 'cmap' not in kwargs.keys(): 
            kwargs['cmap'] = 'jet'
        if 'levels' not in kwargs.keys():
            kwargs['levels'] = 100
            
        plt.tricontourf(x,y,z,**kwargs)


    @staticmethod
    def _scatter_plot(x,y,z,**kwargs):
        if 'cmap' not in kwargs.keys(): 
            kwargs['cmap'] = 'jet'
        if 'levels' not in kwargs.keys():
            kwargs['levels'] = 100
            
        plt.scatter(x,y,s=0.5,c=z,**kwargs)


    def plot_2D(self,plot_type,x:torch.Tensor,net:torch.nn.Module,input_vars,output_var,*,title = None,save_name = None,epoch = None,show = True,domain =None,aspect_ratio = 'equal',**kwargs):
        assert len(input_vars) == 2, 'input vars must be of length 2'
        input_dict,output_dict = self.format(x,net)
        epoch_str = f"at epoch {epoch}" if epoch is not None else ''
        title = f'{plot_type} plot of {output_var} {epoch_str}' if title is None else title 

        plt.cla()
        plt.clf()
        
        if plot_type == 'contour':
            self._contour_plot(input_dict[input_vars[0]],input_dict[input_vars[1]],output_dict[output_var],**kwargs)
        elif plot_type == 'scatter':
            self._scatter_plot(input_dict[input_vars[0]],input_dict[input_vars[1]],output_dict[output_var],**kwargs)


        if domain is not None:
            inv_shape = domain.Domain.symmetric_difference(box(*domain.Domain.bounds))
            if inv_shape.is_empty is not True:
                p = gpd.GeoSeries(domain.Domain.symmetric_difference(box(*domain.Domain.bounds)))
                p.plot(ax = plt.gca(),color = 'white')

        plt.title(title)
        plt.gca().set_aspect(aspect_ratio)
        plt.colorbar()
        plt.xlabel(input_vars[0])
        plt.ylabel(input_vars[1])

        if save_name is not None:
            plt.savefig(save_name)
        
        if show:
            plt.show()
        
        plt.clf()
        plt.cla()
    def set_contour_points(self,domain,ground_truth = None,resolution = 100):
        self.domain = domain
        bounds = domain.bounds
        x_range = (bounds[0],bounds[2])
        y_range = (bounds[1],bounds[3])
        X,Y = torch.meshgrid([torch.linspace(*x_range,resolution),torch.linspace(*y_range,resolution)])
        x,y = X.flatten(),Y.flatten()
        
        xy = torch.stack([x,y],dim = -1)
        #Check if in geometry:
        ps = points(xy.numpy())
        contained = domain.contains(ps)
        self.contour_points = xy[contained]

    def validate_contour(self,net,input_vars,output_var,*,title=None,save_name = None,epoch = None,show = True,**kwargs):
        self.plot_2D('contour',self.contour_points,net,input_vars,output_var,title=title,save_name =save_name,epoch =epoch,show =show,domain = self.domain,**kwargs)


    def validate_scatter(self,net,input_vars,output_var,*,title=None,save_name = None,epoch = None,show = True,**kwargs):
        self.plot_2D('scatter',self.contour_points,net,input_vars,output_var,title=title,save_name =save_name,epoch =epoch,show =show,domain = self.domain,**kwargs)

