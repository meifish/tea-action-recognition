# -*- coding: utf-8 -*-
# phoenixyli 李岩 @2020-04-02 17:10:52

import torch


class Identity(torch.nn.Module):
    """Identity module
    
    x = x
    """

    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):
    """
    """

    # def __init__(self, consensus_type, dim=1):
    #     self.consensus_type = consensus_type
    #     self.dim = dim
    #     self.shape = None

    # def forward(self, input_tensor):
    #     #import pdb; pdb.set_trace()
    #     self.shape = input_tensor.size()
    #     if self.consensus_type == 'avg':
    #         output = input_tensor.mean(dim=self.dim, keepdim=True)
    #     elif self.consensus_type == 'identity':
    #         output = input_tensor
    #     else:
    #         output = None

    #     return output

    @staticmethod
    def forward(ctx, input_tensor, consensus_type, dim=1):
        #import pdb; pdb.set_trace()
        ctx.shape = input_tensor.size()
        ctx.dim = dim
        ctx.consensus_type = consensus_type

        if ctx.consensus_type == 'avg':
            output = input_tensor.mean(dim=ctx.dim, keepdim=True)
        elif ctx.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output


    # def backward(self, grad_output):
    #     if self.consensus_type == 'avg':
    #         grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
    #     elif self.consensus_type == 'identity':
    #         grad_in = grad_output
    #     else:
    #         grad_in = None

    #     return grad_in
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_output_clone = grad_output.clone()
        
        if ctx.consensus_type == 'avg':
            grad_in = grad_output_clone.expand(ctx.shape) / float(ctx.shape[ctx.dim])
        elif ctx.consensus_type == 'identity':
            grad_in = grad_output_clone
        else:
            grad_in = None

        return grad_in, None, None



class ConsensusModule(torch.nn.Module):
    """
    """

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim


    def forward(self, input):
        return SegmentConsensus.apply(input, self.consensus_type, self.dim)
    