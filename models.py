import torch
from cplx import Cplx, modrelu, linear_naive


class AlexNet(nn.Module):
    def __init__(self, in_channel=2, classes=2):
        super(AlexNet, self).__init__()

        self.seq = nn.Sequential(
		    nn.Conv1d(in_channel, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
		    nn.MaxPool1d(kernel_size=3, stride=2),
		    nn.Conv1d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
		    nn.MaxPool1d(kernel_size=3, stride=2),
		    nn.Conv1d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
		    nn.Conv1d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
		    nn.Conv1d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
		    nn.MaxPool1d(kernel_size=3, stride=2),
		    nn.Flatten(),
		    #nn.Dropout(p=0.5),
		    nn.Linear(6400, 4096), nn.ReLU(),
		    #nn.Dropout(p=0.5),
		    nn.Linear(4096, 4096), nn.ReLU(),
		    nn.Linear(4096, classes))

    def forward(self, x):
    	return self.seq(x)



class AlexNet_Complex(nn.Module):
    def __init__(self, in_chaColnnel=1, classes=2):
        super(AlexNet_Complex, self).__init__()

        self.seq = nn.Sequential(
        	#cplxnn.CplxAdaptiveModReLU()
		    cplxnn.Conv1d(in_channel, 96, kernel_size=11, stride=4, padding=1), cplxnn.ReLU(),
		    cplxnn.Conv1d(96, 96, kernel_size=3, stride=2), # In place of pooling
		    cplxnn.Conv1d(96, 256, kernel_size=5, padding=2), cplxnn.ReLU(),
		    cplxnn.Conv1d(256, 256, kernel_size=3, stride=2),
		    cplxnn.Conv1d(256, 384, kernel_size=3, padding=1), cplxnn.ReLU(),
		    cplxnn.Conv1d(384, 384, kernel_size=3, padding=1), cplxnn.ReLU(),
		    cplxnn.Conv1d(384, 256, kernel_size=3, padding=1), cplxnn.ReLU(),
		    #nn.Conv1d(256, 256, kernel_size=3, stride=2),
		    cplxnn.Flatten(),
		    #nn.Dropout(p=0.5),
		    cplxnn.Linear(6400, 4096), cplxnn.ReLU(),
		    #nn.Dropout(p=0.5),
		    cplxnn.Linear(4096, 4096), cplxnn.ReLU(),
		    cplxnn.Linear(4096, classes))

    def forward(self, x):
    	# convert to complex duck typing beforehand
    	# cplx.Cplx.from_numpy(np.array) (1 in channel)
    	return self.seq(x)


