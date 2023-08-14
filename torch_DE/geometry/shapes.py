from shapely.geometry import Polygon,Point

def Circle(center:tuple,r:float,num_points = 1024):
    return Point(center).buffer(r,num_points)

def Rectangle(coords,create_from = 'corners'):
    if create_from == 'corners':
            #[(x1,y1),(x2,y2)]
        (x1,y1),(x2,y2) = coords
        coords = [(x1,y1),(x1,y1+y2),(x2,y2),(x1+x2,y1)]
    elif create_from == 'midpoint':
        # [(x,y),w,h]
        (x,y),w,h = coords
        coords = [(x-w/2,y-h/2),(x-w/2,y+h/2),(x+w/2,y+h/2),(x+w/2,y-h/2)]
    return Polygon(coords)