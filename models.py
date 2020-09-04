import torch
#from cplxmodule.cplxmodule.cplx import Cplx, modrelu, linear_naive

import cplxmodule.nn as cplxnn
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, in_channel=2, classes=2):
        super(AlexNet, self).__init__()

        flatten_dim = 6144

        self.seq = nn.Sequential(
		    nn.Conv1d(in_channel, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
		    nn.Conv1d(96, 96, kernel_size=3, padding=1, stride=2), nn.ReLU(), # MP 1
		    #nn.MaxPool1d(kernel_size=3, stride=2),
		    nn.Conv1d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
		    #nn.MaxPool1d(kernel_size=3, stride=2),
		    nn.Conv1d(256, 256, kernel_size=5, padding=2, stride=2), nn.ReLU(), # MP 2
		    nn.Conv1d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
		    nn.Conv1d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
		    nn.Conv1d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
		    nn.MaxPool1d(kernel_size=3, stride=2),
		    nn.Flatten(),
		    nn.Dropout(p=0.5),
		    nn.Linear(flatten_dim, 4096), nn.ReLU(),
		    nn.Dropout(p=0.5),
		    nn.Linear(4096, 4096), nn.ReLU(),
		    nn.Linear(4096, classes))

    def forward(self, x):
    	return self.seq(x)



class AlexNet_Complex(nn.Module):
    def __init__(self, in_channel=1, classes=2):
        super(AlexNet_Complex, self).__init__()

        flatten_dim = 12800

        self.complex_backbone = cplxnn.CplxSequential(
        	#cplxnn.CplxAdaptiveModReLU()
		    cplxnn.CplxConv1d(in_channel, 96, kernel_size=11, stride=4, padding=1), cplxnn.CplxModReLU(),
		    cplxnn.CplxConv1d(96, 96, kernel_size=3, padding=1, stride=2), cplxnn.CplxModReLU(), # In place of pooling
		    cplxnn.CplxConv1d(96, 256, kernel_size=5, padding=2), cplxnn.CplxModReLU(),
		    cplxnn.CplxConv1d(256, 256, kernel_size=3, padding=1, stride=2), cplxnn.CplxModReLU(),
		    cplxnn.CplxConv1d(256, 384, kernel_size=3, padding=1), cplxnn.CplxModReLU(),
		    cplxnn.CplxConv1d(384, 384, kernel_size=3, padding=1), cplxnn.CplxModReLU(),
		    cplxnn.CplxConv1d(384, 256, kernel_size=3, padding=1), cplxnn.CplxModReLU(),
		    #nn.Conv1d(256, 256, kernel_size=3, stride=2),
		    #nn.Flatten(),
		    cplxnn.CplxToCplx[torch.nn.Flatten](start_dim=-2),
		    cplxnn.CplxDropout(p=0.5),
		    cplxnn.CplxLinear(flatten_dim, 4096), cplxnn.CplxModReLU(),
		    cplxnn.CplxDropout(p=0.5),
		    cplxnn.CplxLinear(4096, 4096), cplxnn.CplxModReLU(),
		    cplxnn.CplxLinear(4096, classes))

        self.seq = nn.Sequential(
		    cplxnn.RealToCplx(),
		    self.complex_backbone,
		    cplxnn.CplxToReal(),
		)

    def forward(self, x):
    	# convert to complex duck type around backbone
    	return self.seq(x)




