import torch

def GradNorm(net:torch.nn.Module,global_weights:torch.Tensor,*losses:torch.Tensor,alpha:float = 0.9,max_weight = None,eps = 1e-5):
    '''
    Gradient Normalisation Method by __ et al
    '''
    assert isinstance(global_weights,torch.Tensor),f'global_weights needs to be of type Torch.Tensor!!! Got {type(global_weights)} type instead'
    assert len(global_weights) == len(losses) , f'the number of weights {len(global_weights)} given do not match the number of losses ({len(losses)})'

    params = [v for v in net.parameters()]
    
    grads = [torch.autograd.grad(loss,params,torch.ones_like(loss),retain_graph=True,allow_unused=True) for loss in losses]

    max_weight = float('inf') if max_weight is None else max_weight
    with torch.no_grad():
        #Get norm
        grad_mags = [torch.sqrt( sum( (torch.sum(g.pow(2))  for g in grad if g is not None))) for grad in grads]
        grad_sum = sum(grad_mags)

        new_weights =torch.tensor([ min(grad_sum/(grad_mag) if grad_mag > eps else global_weight ,max_weight) for grad_mag,global_weight in zip(grad_mags,global_weights)])
        
        #Just to be sure make everything a tensor (we dont call this often so it should be fine)
        return alpha*global_weights + (1-alpha)*new_weights


def Causal_weighting(loss,eps=1.0):
    '''
    Causal training algorithim by _ et al.

    Note this function assumes that the x tensor has been sorted from ascending order in time from t0 to t1 (so that the residual loss are also ordered in time)
    
    Returns: Causal weights for given residual loss
    '''
    with torch.no_grad():
        res_loss = [ l for group in loss['Residual'].values() for l in group.values() ]
        res_loss = sum(res_loss)/len(res_loss)
        res_cumsum = torch.cumsum(res_loss,dim = 0)
        causal_weights = torch.exp(-eps*res_cumsum)
        causal_weights[0] = 1
    return causal_weights
