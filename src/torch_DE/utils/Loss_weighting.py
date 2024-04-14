import torch

def GradNorm(net:torch.nn.Module,global_weights:torch.Tensor,*losses:torch.Tensor,alpha:float = 0.9):
    
    assert isinstance(global_weights,torch.Tensor),f'global_weights needs to be of type Torch.Tensor!!! Got {type(global_weights)} type instead'
    assert len(global_weights) == len(losses) , f'the number of weights {len(global_weights)} given do not match the number of losses ({len(losses)})'

    params = [v for v in net.parameters()]
    
    grads = [torch.autograd.grad(loss,params,torch.ones_like(loss),retain_graph=True,allow_unused=True) for loss in losses]

    with torch.no_grad():
        #Get norm
        grad_mags = [torch.sqrt( sum( (torch.sum(g.pow(2))  for g in grad if g is not None))) for grad in grads]
        grad_sum = sum(grad_mags)

        new_weights =torch.tensor([grad_sum/grad_mag for grad_mag in grad_mags])
        
        #Just to be sure make everything a tensor (we dont call this often so it should be fine)
        return alpha*global_weights + (1-alpha)*new_weights
