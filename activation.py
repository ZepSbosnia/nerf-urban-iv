import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 

class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))
    

    #DP !!!

class _trunc_DP_sigmoid_mod(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return 5 * (torch.sigmoid(x) **10)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        sigmoid_derivative = torch.sigmoid(x) * (1 - torch.sigmoid(x))
        grad_input = g * 50 * sigmoid_derivative * (torch.sigmoid(x) ** 9)
        return grad_input


class _trunc_DP_sigmoid(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (torch.sigmoid(x) )

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        s = torch.sigmoid(x)
        # d/dx sigmoid(x) = s * (1 - s)
        return g * s * (1 - s)


class _SelectiveSigmoid_DP(Function):
    # sigmoid(5x), computed with torch.sigmoid to avoid exp overflow (inf/inf -> NaN) for very negative x
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    def forward(ctx, input):
        output = torch.sigmoid(5 * input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        s, = ctx.saved_tensors
        # d/dx sigmoid(5x) = 5 * s * (1 - s)
        return 5 * s * (1 - s) * grad_output

# Scale and shift function to map sigmoid output to [0, 5]
def scale_and_shift(output):
    return 5 * output  # Adjust this as needed to fit within [0, 5]
    
trunc_exp = _trunc_exp.apply
trunc_DP_sigmoid = _trunc_DP_sigmoid.apply

selective_sigmoid_DP = _SelectiveSigmoid_DP.apply