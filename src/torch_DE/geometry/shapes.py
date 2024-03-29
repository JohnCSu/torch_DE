from shapely.geometry import Polygon,Point, LineString
from .sampling import *
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import torch
from numpy.random import rand
def Circle(center:tuple,r:float,num_points = 1024):
    return Point(center).buffer(r,num_points)

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
    def __init__(self,base :Polygon ) -> None:
        self.operations ={'base_op_1':base}
        self.num_operations = 1 
        self.Domain = base
        self.bounds = self.Domain.bounds
        self.boundary = self.Domain.boundary
        self.contains = self.Domain.contains
        self.points = None
        self.triangles = None
        #Create Lines from bounds
        self.boundary_groups = {}
        self.create_domain_exterior_edges()
    
    def __getitem__(self,key):
        #Returns the coords of a part of the domain
        return self.operations[key]
    
    def create_domain_exterior_edges(self):
        p = self.boundary.coords
        num_p = len(p)

        self.boundary_groups.update(
            {
                f'exterior_edge_{i}':  LineString([p[i],p[i+1]]) for i in range(num_p-1)
            }

        )



    def merge(self,*shapes,names = None):
        self.boolean_op(*shapes,names = names,op = 'add',inplace=True)
    def remove(self,*shapes,names= None):
        self.boolean_op(*shapes,names=names,op = 'sub',inplace=True)

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
    
    def add_boundary_group(self,shapeID,name = None):
        if name is None:
            key = shapeID
        else:
            key = name
        self.boundary_groups[key] = self.operations[shapeID].exterior

    def clear_mesh(self):
        self.points,self.triangles = (None,None)
        print('mesh cleared')

    def generate_points(self,n,shapeID = None,func = None,output_type = 'torch',**kwargs):
        '''
        sample points from domain. Default triangulates the domain and then samples from the
            triangulated domain or specified Group Shape.
        
        Users can implement their own custom function by passing it in with func. func 
        should take the form f(n,shape,kw1,kw2...) where n is the number of points to 
        sample, shape is some Polygon object followed by any other keyword arguements
        
        '''
        if shapeID is None:
            shape = self.Domain
        else:
            shape = self.operations[shapeID]
        
        if func is None:
            if self.points is None:
                points,triangles = triangulate_shape(shape,**kwargs)
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