from shapely.geometry import Polygon,Point, LineString
from .sampling import *
from scipy.spatial import Delaunay

from typing import Callable
import matplotlib.pyplot as plt
import torch
from numpy.random import rand
from torch_DE.utils import RegularGridInterpolator
import geopandas as gdp

def Circle(center:tuple,r:float,num_points = 1024):
    return Point(center).buffer(r,resolution = num_points//4)

def Rectangle(coords: list,create_from = 'corners'):
    '''
    Create a Rectangle using either the corners method or midpoint method

    if create_from == 'corners':
        coords = [(x1,y1),(x2,y2)] where (x1,y1) is the coords for the lower left corner and (x2,y2) is the coord for the upper right corner
    
    if create_from == 'midpoint':
        coord = [(x,y),(w,h)] where (x,y) is the centre of the rectangle and (w,h) is the width and height of the rectangle respectively

    returns:
        Shapely Polygon Object representing the rectangle created
    '''

    if create_from == 'corners':
            #[(x1,y1),(x2,y2)]
        (x1,y1),(x2,y2) = coords
        coords = [(x1,y1),(x1,y2),(x2,y2),(x2,y1)]
    elif create_from == 'midpoint':
        # [(x,y),(w,h)]
        (x,y),(w,h) = coords
        coords = [(x-w/2,y-h/2),(x-w/2,y+h/2),(x+w/2,y+h/2),(x+w/2,y-h/2)]
    return Polygon(coords)




class Domain2D():
    '''
    Domain Object to create shapes from. Because of how Polygon classes work in shapely, subclassing is weird. It seems better to treat shapely objects 
    functionally than OOB as it is immutable
    '''
    def __init__(self,base :Polygon,is_rectangle = False ) -> None:
        self.operations ={'base_op_1':base}
        self.num_operations = 1 
        self.Domain = base

        self.is_rectangle = is_rectangle
        self.bounds = self.Domain.bounds
        self.boundary = self.Domain.boundary
        self.contains = self.Domain.contains
        self.points = None
        self.triangles = None
        #Create Lines from bounds
        self.boundary_groups = {}
        self.create_domain_exterior_edges()
        self.sdf = None
        
        self.partitions = Partition_Group()
    def __getitem__(self,key):
        #Returns the coords of a part of the domain
        return self.operations[key]
    

    def partition_2points(self,p1,p2,name):
        self.partitions.partition_2Points(self,p1,p2,name)

    def create_domain_exterior_edges(self):
        p = self.boundary.coords
        num_p = len(p)
        if self.is_rectangle:
            wall_names = ['left wall','top wall','right wall','bot wall']
            self.boundary_groups.update(
            {
                wall_name:  (LineString([p[i],p[i+1]]),'linear')  for i,wall_name in enumerate(wall_names) 
            }
        )
        else:
            self.boundary_groups.update(
                {
                    f'exterior_edge_{i}':  (LineString([p[i],p[i+1]]),'linear')  for i in range(num_p-1) 
                }

            )
    def merge(self,*shapes,names = None):
        self.boolean_op(*shapes,names = names,op = 'add',inplace=True)
    def remove(self,*shapes,names= None):
        self.boolean_op(*shapes,names=names,op = 'sub',inplace=True)

    def create_sdf(self,resolution: int = 256,scale_factor = 1.,device = 'cpu'):
        '''
        Creates a slightly modified SDF function for the current geometry. The SDF sets the values of points outside the geometry to zero tahter than be negative.

        Note that this method uses a burte force approach using Shapely distance function. We create an aprroximate SDF from a uniform grid of 10,000 points \n
        plus 2000 points from the boundary and then use scipy LinearNDInterpolator to create a interp field.

        
        Input:
            resolution: int (default = 256) Number of grid points in each X,Y direction to take
            scale_factor: float or str. If `scale_factor == 'normalize'` then the sdf is scaled by the max(sdf) otherwise SDF is scaled by `scale_factor*SDF`. Default is 1
            device: str (default = 'cpu') Device to put the grid on. Default cpu but can be set to cuda
        Output:
            SDF func(xy) where xy is a tensor of shape (N,2). The output of this function is a tensor of size (N) of the signed distances of each point.
        '''
        #We use a brute force method for SDF. Points outside the domain are set to zero. This is good enough for PINN applications
        xmin,ymin,xmax,ymax = self.Domain.bounds
        x,y = [torch.linspace(xmin,xmax,resolution),torch.linspace(ymin,ymax,resolution)]
        X,Y = torch.meshgrid(x,y,indexing = 'ij')
        xg,yg = X.flatten(),Y.flatten()

        points = [Point(x1,y1) for x1,y1 in zip(xg,yg)]
        #Brute Force SDF
        distance = torch.tensor(self.Domain.boundary.distance(points))
        distance[~self.contains(points)] *= 0
        
        distance = distance.reshape((resolution,resolution)).to(device = device)

        distance = distance/distance.max() if scale_factor == 'normalize' or scale_factor == 'normalise' else distance*scale_factor
        self.sdf = RegularGridInterpolator((x,y),distance)
        self.sdf.set_device(device)

        return self.sdf

    def plot_sdf(self):
        if self.sdf is None:
            raise ValueError('SDF has not been defined yet!')
        
        device = self.sdf.device

        self.sdf.set_device('cpu')
        xmin,ymin,xmax,ymax = self.Domain.bounds
        X,Y = torch.meshgrid(torch.linspace(xmin,xmax,100),torch.linspace(ymin,ymax,100))
        x,y = X.flatten(),Y.flatten()

        a = torch.stack((x,y),dim = -1).cpu()
        z = self.sdf(a)
        plt.gca().set_aspect('equal')
        plt.tricontourf(x.cpu(),y.cpu(),z.cpu(),levels = 100,cmap = 'jet')
        plt.title('SDF of Domain (points outside domain map to zero)')
        plt.colorbar()
        plt.show()
        self.sdf.set_device(device)

    def boolean_op(self,*shapes,names=None,op = None,inplace = False):
        '''
        Operation to merge or remove shapes from existing combined into one as only difference is whether union or difference is called
        '''
        if op == 'add':
            oper = 'union'
        elif op == 'sub':
            oper = 'difference'

        if names is not None:
            assert len(shapes) == len(names)
        
        looper = shapes if names is None else zip(shapes,names)

        for val in looper:
            if names is None:
                shape = val                
            else:
                shape,ID = val
            
            # print(shape)
            if isinstance(shape,Domain2D):
                shape = shape.Domain

            self.Domain = getattr(self.Domain,oper)(shape)
            #Update bounds
            self.bounds = self.Domain.bounds
            self.contains = self.Domain.contains
            
            self.num_operations += 1
            
            if names is None:
                ID = f'{op}_{str(shape.geom_type)}_op_{self.num_operations}'  
            self.operations[ID] = shape
        
        if not inplace:
            return self
    
    def add_boundary_group(self,shapeID,line_type,name = None):
        '''
        Add boundary group to domain
        
        shapeID: str name of shape operation previously performed
        line_type: str['curve','linear'] Chose curved when defining a boundary with curves. choose linear when the boundary contains only straight lines
        '''
        if name is None:
            key = shapeID
        else:
            key = name
        
        assert line_type == 'curve' or line_type == 'linear', 'line_type can only be strings curve or linear'

        self.boundary_groups[key] = (self.operations[shapeID].boundary,line_type)

    
    def generate_points_from_boundary(self,boundary,points_per_line = 100,random = False):
            exterior,exterior_type = self.boundary_groups[boundary]
            if exterior_type == 'curve':
                return torch.tensor(exterior.coords)
            elif exterior_type == 'linear':
                num_lines = len(exterior.coords) - 1
                return self.generate_points_from_line(exterior,points_per_line*num_lines,random = random)


    def generate_boundary_points(self,num_points = 100, random = False):
        return   {name:self.generate_points_from_boundary(name,num_points,random) for name in self.boundary_groups.keys()} 

    
    def clear_mesh(self):
        self.points,self.triangles = (None,None)
        print('mesh cleared')

    

    def generate_points(self,n:int,shapeID:str = None,func:Callable = None,output_type:str = 'torch',seed:int = None,**kwargs):
        '''
        sample points from domain. Default triangulates the domain and then samples from the
            triangulated domain or specified Group Shape.
        
        Users can implement their own custom function by passing it in with func. func 
        should take the form f(n,shape,kw1,kw2...) where n is the number of points to 
        sample, shape is some Polygon object followed by any other keyword arguements
        
        '''
        if seed is not None:
            np.random.seed(seed)
            
        if shapeID is None:
            shape = self.Domain
        else:
            shape = self.operations[shapeID]
        
        if func is None:
            if self.points is None:
                points,triangles = triangulate_shape(shape,**kwargs)
                #Cache the mesh
                self.points,self.triangles=points,triangles 
            else:
                points,triangles = self.points,self.triangles
            out = generate_points_from_triangles(points,triangles,n)
        else:
            out =  func(n,shape,**kwargs)

        if output_type == 'numpy':
            return out
        elif output_type == 'torch':
            return torch.tensor(out).to(torch.float32)


    @staticmethod
    def generate_points_from_line(line,num_points,random = True):
        '''
        Generate points on a line

        shape: LineString | a list of two points in the form [[x1,y1],[x2,y2]]
        '''
        if random:
            gen_points = rand(num_points)
        else:
            gen_points = np.linspace(0,1,num_points)

        return torch.tensor([line.interpolate(d,normalized=True).coords[0] for d in gen_points ] )
    @staticmethod
    def generate_points_between_two_points(end_points,num_points,random = True):
        line = LineString(end_points)
        return Domain2D.generate_points_from_line(line,num_points,random)


    def plot(self,exterior= False,partitions = True,aspect_ratio = 'equal', **kwargs):
        plotter = gdp.GeoSeries(self.Domain)
        p_groups = gdp.GeoSeries([v[0] for v in self.partitions.values()])
        fig, ax = plt.subplots()
        
        if exterior:
            plotter.boundary.plot(ax=ax,**kwargs)
            if partitions:
                p_groups.plot(ax=ax)
        else:
            plotter.plot(ax=ax,**kwargs)

        ax.set_aspect(aspect_ratio)

        plt.show()
        plt.close(fig) 

def is_tri_in_shape(points,triangles,shape):
    for triangle in triangles:
        #I have an array of the points [p1,p2,p3] which is the index
        tri_points = points[triangle]
        tri = Polygon(tri_points)
        yield shape.contains(tri) 


def remove_tri_holes(triangulation,shape):
    points = triangulation.points
    triangles = triangulation.simplices

    tri_mask = [tri_check for tri_check in is_tri_in_shape(points,triangles,shape)]
    return triangles[tri_mask]



def triangulate_shape(shape,show_plot = False):
    '''
    Triangulate an arbitary shape using Delaunay Alogorithim from SciPy.

    Args:
    shape: Polygon | Domain2D -> Shape;y object to triangulate
    show_plot: bool (Default : False) -> display the triangulated shapes

    Returns:
    points : np.array -> array of (x,y) co-ordinates of each vertex in the shape of size (N,2)
    triangles : np.array -> an (N,3) array. Each row indicates the index of the 3 points in the points array that makeup the triangle

    Example: triangles[0] = [0,2,3] means that the 1st triangle made from points[0],points[2] and points[3]
    '''
    if isinstance(shape,Domain2D):
        shape =shape.Domain
    
    vertices = list(shape.exterior.coords)
    holes = [list(hole.coords) for hole in shape.interiors]
    all_vertices = np.array(vertices + [v for hole in holes for v in hole])
    # Create the Delaunay triangulation
    triangulation = Delaunay(all_vertices)

    #Find all triangles that is contained within the shape
    triangles = remove_tri_holes(triangulation,shape)
    points = triangulation.points
    #PLot Results. Note avoid plotting circles (these are technically a high n-sided polygon)
    if show_plot:        
        # Prepare data for plotting
        plt.triplot(all_vertices[:,0], all_vertices[:,1], triangles)
        plt.plot(all_vertices[:,0], all_vertices[:,1], 'o')
        plt.show()

    return points,triangles        




class Partition_Group(dict):
    def __init__(self):
        super().__init__()
    def partition_2Points(self,domain:Domain2D,p1,p2,name):
        line = LineString((p1,p2))
        self[name] = (domain.Domain.intersection(line),'linear')

    def partition_from_lines(self,domain,lines,name):
        self[name] = (domain.Domain.intersection(lines),'linear')

    def partition_from_curve(self,domain,line,name):
        self[name] = (domain.Domain.intersection(line),'curve')


    def _gen_points(self,partition_name,points_per_line,random = False):
        line,line_type = self[partition_name]
        if line_type == 'curve':
            return torch.tensor(line.coords)
        elif line_type == 'linear':
            # num_lines = len(line.coords) - 1
            return Domain2D.generate_points_from_line(line,points_per_line,random = random)


    def generate_points(self,num_points=100,random = False):
        return {k: self._gen_points(k,num_points,random) for k in self.keys()}
