import torch.nn as nn
class Operator_Module(nn.Module):
    def __init__(self):
        super().__init__(self)
        pass


class DeepONet(Operator_Module):
    def __init__(self,branch_net : nn.Module, trunk_net:nn.Module):
        super().__init__(self)

        self.branch_net = branch_net
        self.trunk_net = trunk_net


    def set_branch(self,x):
        self.b = self.branch_net(x)

    def query(self,x):
        self.b*self


    def forward(self,x):
        pass


