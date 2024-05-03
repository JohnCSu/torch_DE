from matplotlib import pyplot as plt
import torch
from typing import Union,Iterable
from shapely import points,box
import geopandas as gpd
from torch_DE.utils.sampling import add_time_point,add_time_col
from numpy.typing import ArrayLike

class Plotter():
    def __init__(self,input_vars:Union[list,tuple],output_vars:Union[list,tuple]) -> None:
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.contour_points = None
        self.domain =None
        self.time = 0
        self.time_interval = None
        self.is_time = False


        self.plotting_kwargs = {
            'cmap':'jet',
            'levels':100
        }

        self.colorbar_kwargs = {}

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

    def add_plotting_kwargs(self,**kwargs):
        '''
        kwargs to add to plotting command e.g. plt.tricontourf(x,y,z,...,**kwargs)
        '''
        self.plotting_kwargs.update(kwargs)


    def add_colorbar_kwargs(self,**kwargs):
        self.colorbar_kwargs.update(kwargs)

    @staticmethod
    def add_domain_to_plot(domain,ax):
        if domain is not None:
            inv_shape = domain.Domain.symmetric_difference(box(*domain.Domain.bounds))
            if inv_shape.is_empty is not True:
                p = gpd.GeoSeries(domain.Domain.symmetric_difference(box(*domain.Domain.bounds)))
                p.plot(ax = ax,color = 'white')



    def plot_2D(self,plot_type:str,x:ArrayLike,y:ArrayLike,z:ArrayLike,*,input_vars = None,output_var= None,output = False,save_name = None,epoch = None,show = True,aspect_ratio = 'equal',title = None,**kwargs):
        '''
        Plot a contour or scatter plot of a 2D Field

        inputs:
            - plot_type: str of either `'contour'` or `'scatter'` to choose the type of plot from
            - x: tensor/array-like representing the coordinates for z data along the x-axis should be a vector of size (N,)
            - y: tensor/array-like representing the coordinates for z data along the y-axis should be a vector of size (N,)
            - z: tensor/array-like for the output of the data.

            
        The following optional Keywords
            - input_vars: list|tuple pair of str determinig the x and y axis respectively. each str must be contained in `self.input_vars`. If None the default labels for each axis is 'x' and 'y' respectively.
            - output_var: str name of z_label to plot. If None the default name for the z data is 'u'
            - output: bool (default `False`) output plotting data as a pair of tuples with the first being the plotting data and the 2nd tuple as the respective labels e.g `(x,y,z),(x_label,y_label,z_label)`.
                 Data is return as numpy arrays. useful if you want more control in your plotting
            - save_name: str | None : filename to save figure. if None the plot is not saved
            - show: bool (default: `True`) boolean on whether to show plot or not
            - aspect_ratio: str option to determine aspect ratio of axes options inlcude 'equal','auto' see `ax.set_aspect` for more details

        To set additional keywords for the plotting command (such as tricontourf or `plt.scatter`), use the `add_colorbar_kwargs` method. For colorbar modification, use the add_colorbar_kwargs `add_colorbar_kwargs` method

        Output:
            If output is set to True then return a pair of tuples with the first being the plotting data and the 2nd tuple as the respective labels e.g `(x,y,z),(x_label,y_label,z_label)`.
        '''
        
        if input_vars is None:
            input_vars = ['x','y']
        if output_var is None:
            output_var = 'u'
        
        epoch_str = f"at epoch {epoch}" if epoch is not None else ''
        time_str = f"at time {self.time}" if self.time is not None else ''
        title = f'{plot_type} plot of {output_var} {time_str} {epoch_str}' if title is None else title 

        if save_name is None and show is False: 
            if output:
                return (x,y,z),(tuple(input_vars) + (output_var,))
            else:
                raise ValueError('At least one variable of either save_name, show or output must be not None or False')
        
        fig,ax = plt.subplots()

        if plot_type == 'contour':
            plotting_mappable = ax.tricontourf(x,y,z,**self.plotting_kwargs)
        elif plot_type == 'scatter':
            plotting_mappable = ax.scatter(x,y,c=z,**self.plotting_kwargs)

        self.add_domain_to_plot(self.domain,ax)

        ax.set_title(title)
        ax.set_aspect(aspect_ratio)
        ax.set_xlabel(input_vars[0]), ax.set_ylabel(input_vars[1])

        fig.colorbar(plotting_mappable,**self.colorbar_kwargs)
        if save_name is not None:
            fig.savefig(save_name)
        if show:
            fig.show()
            
        
        plt.close(fig)

        if output:
            return (x,y,z),(tuple(input_vars) + (output_var,))


    def set_contour_points(self,domain,resolution = 100,time_interval = None,time_col = -1):
        self.domain = domain
        bounds = domain.bounds
        x_range = (bounds[0],bounds[2])
        y_range = (bounds[1],bounds[3])

        x,y =  [ grid.flatten() for grid in torch.meshgrid([torch.linspace(*x_range,resolution),torch.linspace(*y_range,resolution)])]
        
        xy = torch.stack([x,y],dim = -1)
        #Check if in geometry and retain point in geometry:
        ps = points(xy.numpy())
        contained = domain.contains(ps)
        self.contour_points = xy[contained]

        if time_interval is not None:
            self.is_time = True
            self.time_interval = time_interval
            self.contour_points = add_time_col(self.contour_points,col = time_col)
            self.time = 0
    def set_time_point(self,t):
        if self.is_time is False:
            raise ValueError(f'time axis must first be invoked when calling set_contour_points')
        else:
            t0,T = self.time_interval
            if t0 <= t <= T:                
                add_time_point(self.contour_points,t)
                self.time = t
                return
            raise ValueError("t is outside of time_interval bounds")


    def contour(self,net,input_vars,output_var,*,output=False,save_name = None,epoch = None,show = True,**kwargs):
        '''
        Plot Contour using points based on defined points from `Plotter().set_contour_points` if you want to change the input x use plot_2D. See `plot_2D()` 
        
        Required Inputs:
            - net: nn.Module | Callable function. The output should be (N,O) where O is the number of elements in `self.output_vars` 
            - input_vars: list|tuple pair of str determinig the x and y axis respectively. each str must be contained in `self.input_vars`
            - output_var: str output_var to plot 

        for more details on optional keywords see `Plotter().plot_2D()`     
        '''
        assert len(input_vars) == 2, 'input vars must be of length 2'
        
        input_dict,output_dict = self.format(self.contour_points,net)
        x,y = input_dict[input_vars[0]],input_dict[input_vars[1]]
        z = output_dict[output_var]

        return self.plot_2D('contour',x,y,z,input_vars = input_vars,output_var = output_var,output=output,save_name =save_name,epoch =epoch,show =show,**kwargs)


    def scatter(self,net,input_vars,output_var,*,output=False,save_name = None,epoch = None,show = True,**kwargs):
        '''
        Plot Scatter using points based on defined points from `Plotter().set_contour_points` if you want to change the input x use plot_2D. See `plot_2D()` for more details in implementation
        
        Required Inputs:
            - net: nn.Module | Callable function. The output should be (N,O) where O is the number of elements in `self.output_vars` 
            - input_vars: list|tuple pair of str determinig the x and y axis respectively. each str must be contained in `self.input_vars`
            - output_var: str output_var to plot. output_var must be contained in the `self.output_vars`

        for more details on optional keywords see `Plotter().plot_2D()`   
        
        '''
        assert len(input_vars) == 2, 'input vars must be of length 2'
        
        
        input_dict,output_dict = self.format(self.contour_points,net)
        x,y = input_dict[input_vars[0]],input_dict[input_vars[1]]
        z = output_dict[output_var]

        return self.plot_2D('scatter',x,y,z,input_vars = input_vars,output_var = output_var,output=output,save_name =save_name,epoch =epoch,show =show,**kwargs)

