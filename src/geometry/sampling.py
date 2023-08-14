from shapely.geometry import Polygon,Point
import numpy as np

def rejection_sampler(n,shape,MAX_ITER = 100):
    '''
    Standard Rejection sampler where one bounds a box around the shape and samples from the box. The point is accepted if it is within the shape otherwise
    rejected.

    This method is ok if the domain area can be bounded well by a rectangle but gets very expensive if other wise. if the shapes Area is A, and the bounding
    box's area is B then the average number of points m one would need to sample n points is m = n/(A/B)
    '''
    #Keep generating size n array
    n_generated = 0
    x1,y1,x2,y2 =shape.bounds
    area = shape.area
    
    #Find fraction of area to bounds area
    prop = area/((x2-x1)*(y2-y1))
    #Someone teach me how to append arrays properly
    generated_list = []
    
    #Number of points to sample m
    m = n/prop
    for _ in range(MAX_ITER):
        random_coords = np.column_stack((
            np.random.uniform(x1, x2, m),
            np.random.uniform(y1, y2, m)))
        
        points = map(Point,random_coords)
        
        valid_points = random_coords[list(map(shape.contains,points))] 
        n_generated += len(valid_points)
        generated_list.append(valid_points)
        if n_generated >= n:
            return np.concatenate(generated_list,axis = 0)[0:n]
        



def triangle_proportion(points,triangles):
    '''
    Return the proportion of total area that each triangle comprises of.

    Args:
    points: np.array of size (p,2) containing a list of all vertices that make up all triangles

    triangles: np.array of size (n,3) containing an array of the points that make up each triangle. 
        Each triangle is the array is expressed as [pi,pj,pk] where pi is the ith point in the points array
    
    Returns:
    probs: np.array os size (n) of the probabilities/proportion of each triangle with respect to the total area of all triangles
    '''
    areas = [Polygon(points[triangle]).area for triangle in triangles]
    total_area = sum(areas)
    probs = [area/total_area for area in areas]
    return np.array(probs)


def sample_triangles(triangles,n,probs):
    '''
    Given an array of triangle points and their corresponding probabilities, sample from this array n times

    Args:
    points: np.array of size (p,2) containing a list of all vertices that make up all triangles

    triangles: np.array of size (n,3) containing an array of the points that make up each triangle. 
        Each triangle is the array is expressed as [pi,pj,pk] where pi is the ith point in the points array
    
    probs: np.array os size (n) of the probabilities/proportion of each triangle with respect to the total area of all triangles. 
        Can be obtained from triangle_proportion function
    '''    
    indices = np.arange(len(triangles))
    return np.random.choice(indices,size = n, p = probs)



def generate_points_from_triangles(points,triangles,n):
    '''
    Given a list of triangles and the coordinates of each vertex that comprise a Domain/polygon shape sample n points

    Reflection step and sampling from triangles is explained by a great blog post by Rick Wicklin:
    https://blogs.sas.com/content/iml/2020/10/19/random-points-in-triangle.html  

    All n points are guranteed to be within the shape with this method.

    Args:
    points: np.array of size (p,2) containing a list of all vertices that make up all triangles

    triangles: np.array of size (n,3) containing an array of the points that make up each triangle. Each triangle is the array is expressed as [pi,pj,pk]
        where pi is the ith point in the points array
    '''

    #Get probabilities of each triangle (based on area) and then sample from the distribution n times
    probs = triangle_proportion(points,triangles)
    indices = sample_triangles(triangles,n,probs)
    
    # Creates an nx3x2 array of points - 2nd axis repr the 3 points of the ith sampled triangle and 3rd axis is coords of this sampled triangle (x,y)
    coords = points[triangles[indices]]

    #Generate points on unit square.
    U = np.random.rand(n,2)
    #Reflection of points of unit square see link for explanation
    U[U[:,0] + U[:,1] > 1 ] = 1-U[U[:,0] + U[:,1] > 1]
    
    #Create a and b vectors:
    a = coords[:,1,:] - coords[:,0,:]
    b = coords[:,2,:] - coords[:,0,:]

    #Generate point in each triangle
    p = a*(U[:,0])[:,np.newaxis] + b*(U[:,1])[:,np.newaxis]+ coords[:,0,:]
    return p 