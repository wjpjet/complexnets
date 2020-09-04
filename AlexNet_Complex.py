import cplxmodule.nn as cplxnn
import torch.nn as nn

class AlexNet_Complex(nn.Module):
    def __init__(self, in_channel=1, classes=2):
        super(AlexNet_Complex, self).__init__()

        flatten_dim = 5888

        self.seq = nn.Sequential(
        	#cplxnn.CplxAdaptiveModReLU()
		    cplxnn.CplxConv1d(in_channel, 96, kernel_size=11, stride=4, padding=1), cplxnn.CplxModReLU(),
		    cplxnn.CplxConv1d(96, 96, kernel_size=3, stride=2), # In place of pooling
		    cplxnn.CplxConv1d(96, 256, kernel_size=5, padding=2), cplxnn.CplxModReLU(),
		    cplxnn.CplxConv1d(256, 256, kernel_size=3, stride=2),
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

    def forward(self, x):
    	# convert to complex duck typing beforehand
    	return self.seq(x)

