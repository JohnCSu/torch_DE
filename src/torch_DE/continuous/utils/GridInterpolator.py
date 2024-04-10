import torch
from itertools import product

'''
This code is sourced from https://github.com/sbarratt/torch_interpolations under the Apache 2.0 License and is modified for 2D linear interpolation
'''

assert hasattr(
    torch, "bucketize"), "Need torch >= 1.7.0; install at pytorch.org"


class RegularGridInterpolator:
    '''
    2D interpolation class. This code is sourced from  https://github.com/sbarratt/torch_interpolations under the Apache 2.0 License with minor modifications
    
    Modifications:
        Method to change device on points and values attributes
        Call method taks in a (N,2) tensor rather than a list of N tensors
        Torch.grad is turned off
    '''
    def __init__(self, points, values):
        self.points = points
        self.values = values

        assert isinstance(self.points, tuple) or isinstance(self.points, list)
        assert isinstance(self.values, torch.Tensor)

        self.ms = list(self.values.shape)
        self.n = len(self.points)

        assert len(self.ms) == self.n

        for i, p in enumerate(self.points):
            assert isinstance(p, torch.Tensor)
            assert p.shape[0] == self.values.shape[i]
        self.device = 'cpu'
    def set_device(self,device):
        '''
        Change the device the points are on. Usual choices are cpu or cuda
        '''
        self.points = [p.to(device = device) for p in self.points]
        self.values = self.values.to(device = device)
        self.device = device
    def __call__(self, xy):
        '''
        Input
            xy a tensor of size (N,D). It assumes that the first 2 D dimensions represent x and y respectively
        '''
        return self.interpolate([xy[:,0],xy[:,1]])
    def interpolate(self,points_to_interp):
        with torch.no_grad():
            assert self.points is not None
            assert self.values is not None
            assert len(points_to_interp) == len(self.points)
            K = points_to_interp[0].shape[0]
            for x in points_to_interp:
                assert x.shape[0] == K

            idxs = []
            dists = []
            overalls = []
            for p, x in zip(self.points, points_to_interp):
                idx_right = torch.bucketize(x, p)
                idx_right[idx_right >= p.shape[0]] = p.shape[0] - 1
                idx_left = (idx_right - 1).clamp(0, p.shape[0] - 1)
                dist_left = x - p[idx_left]
                dist_right = p[idx_right] - x
                dist_left[dist_left < 0] = 0.
                dist_right[dist_right < 0] = 0.
                both_zero = (dist_left == 0) & (dist_right == 0)
                dist_left[both_zero] = dist_right[both_zero] = 1.

                idxs.append((idx_left, idx_right))
                dists.append((dist_left, dist_right))
                overalls.append(dist_left + dist_right)

            numerator = 0.
            for indexer in product([0, 1], repeat=self.n):
                as_s = [idx[onoff] for onoff, idx in zip(indexer, idxs)]
                bs_s = [dist[1 - onoff] for onoff, dist in zip(indexer, dists)]
                numerator += self.values[as_s] * torch.prod(torch.stack(bs_s), dim=0)
            denominator = torch.prod(torch.stack(overalls), dim=0)
            return numerator / denominator
    
        
        


if __name__ == '__main__':
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    points = [torch.arange(-.5, 2.5, .2) * 1., torch.arange(-.5, 2.5, .2) * 1.]
    values = torch.sin(points[0])[:, None] + 2 * torch.cos(points[1])[None, :] + torch.sin(5 * points[0][:, None] @ points[1][None, :])
    gi = RegularGridInterpolator(points, values)

    X, Y = np.meshgrid(np.arange(-.5, 2.5, .02), np.arange(-.5, 2.5, .01))
    points_to_interp = [torch.from_numpy(
        X.flatten()).float(), torch.from_numpy(Y.flatten()).float()]
    fx = gi(points_to_interp)
    print(fx)


    resolution = 100
    xmin,ymin,xmax,ymax = (0,0,0.1,0.1)
    

    x = torch.linspace(xmin,xmax,resolution)
    y = torch.linspace(ymin,ymax,resolution)
    X,Y = torch.meshgrid(x,y)
    xg,yg = X.flatten(),Y.flatten()

    points = [1 for x1,y1 in zip(xg,yg)]
    #Brute Force SDF
    distance = torch.tensor(points)
    # distance[~self.contains(points)] *= 0
    
    distance = distance.reshape((resolution,resolution))

    sdf_field = RegularGridInterpolator((x,y),distance)
    # fig, axes = plt.subplots(1, 2)

    # axes[0].imshow(np.sin(X) + 2 * np.cos(Y) + np.sin(5 * X * Y))
    # axes[0].set_title("True")
    # axes[1].imshow(fx.numpy().reshape(X.shape))
    # axes[1].set_title("Interpolated")
    # plt.show()