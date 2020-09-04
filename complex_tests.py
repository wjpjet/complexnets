import torch

# complex valued tensor class
from cplxmodule import cplx

# converters
from cplxmodule.nn import RealToCplx, CplxToReal

# layers of encapsulating other complex valued layers
from cplxmodule.nn import CplxSequential

# common layers
from cplxmodule.nn import CplxConv1d, CplxLinear

# activation layers
from cplxmodule.nn import CplxModReLU

from AlexNet_Complex import AlexNet_Complex


################################

n_features, n_channels = 16, 4
z = torch.randn(3, n_features*2)

cplx = RealToCplx()(z)
print(cplx)


