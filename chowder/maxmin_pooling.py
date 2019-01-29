import torch
from torch.autograd import Function
import torch.nn as nn


from torch.autograd import Function

class MaxMinPoolingFunction(Function):
    """MinMax Layer for pooling the top and bottom instances of a given 1D embedding (section 2.3)"""
    def __init__(self, R):
        """
            Args:
                R: Number of top and bottom instances to pool after convolving
        """
        super().__init__()
        self.R = R
        
    
    def forward(self, x_input):
        """Propogates forward through the network given the 1D conv layer

            Args: 
                x_input: Feature embedding from a 1D conv layer 
                         of shape (batch_size, 1, n_regionss)
            
            Returns:
                output: top and bottom instances of shape (batch_size, 2 * R)
        """
        batch_size = x_input.shape[0]
        n_regions = x_input.shape[2]
        
        # sort the feature embedding and save the max & min indices for backprop
        x_sorted, x_indices = torch.sort(x_input.view(batch_size, n_regions), 
                                         dim=1, descending=True)

        self.indices_max = x_indices.narrow(1, 0, self.R)
        self.indices_min = x_indices.narrow(1, -self.R, self.R)

        output_max = x_sorted.narrow(1, 0, self.R)
        output_min = x_sorted.narrow(1, -self.R, self.R)
        
        # concat the top & bottom instances for a MLP classifier (Figure 2)
        output = torch.cat((output_max, output_min), dim=1)
        
        self.save_for_backward(x_input)
        return output.view(batch_size, 2*self.R)
    
    
    def backward(self, grad_output):
        """Propogates back through the network returning the gradients only through
        the tiles selected as the top & bottom instances
        
            Args:
                grad_output: tensor of shape (batch_size, 2*R) storing the gradients 
                             from the previous layer(s)
            
            Returns:
                grad_input: tensor of shape (batch_size, 1, n_regions) such that all
                            gradient are 0'd out except indices that stored the top &
                            bottom instances
        """
        
        x_input, = self.saved_tensors
        batch_size = x_input.shape[0]
        n_regions = x_input.shape[2]
        
        # store the gradients from the max * min R entries
        grad_output_max = grad_output.narrow(1, 0, self.R)
        grad_output_min = grad_output.narrow(1, -self.R, self.R)
        
        # 0 out all other gradients not used as a top/bottom instance
        grad_input_max = grad_output.new()
        grad_input_max.resize_(batch_size, n_regions).fill_(0)
        grad_input_max.scatter_(1, self.indices_max, grad_output_max)
        
        grad_input_min = grad_output.new()
        grad_input_min.resize_(batch_size, n_regions).fill_(0)
        grad_input_min.scatter_(1, self.indices_min, grad_output_min)
        
        # collapse both tensors by adding them along the tile axis
        grad_input = grad_input_max.add(grad_input_min)
        return grad_input.view(batch_size, 1, n_regions)


    
class MaxMinPooling(nn.Module):
    """MaxMin Module to add to our network such that pytorch can simply call forward"""
    def __init__(self, R):
        super().__init__()
        self.R = R
        
    
    def forward(self, x):
        x = MaxMinPoolingFunction(self.R)(x)
        return x
    
    
    def __repr__(self):
        return '{0} (R={1})'.format(self.__class__.__name__, self.R)