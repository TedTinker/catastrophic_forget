import torch
from torch import nn
from torch.optim import Adam
from blitz.modules import BayesianLinear

from get_data import x_shape

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

def init_weights(m):
    try:
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except:
        pass

class one_cnn(nn.Module):
    def __init__(self, in_channels, out_channels, pool):
        super().__init__()
        
        self.cnn = nn.Sequential(
            ConstrainedConv2d(
                in_channels = in_channels, 
                out_channels = out_channels, 
                kernel_size = 3, 
                padding=1, 
                padding_mode='reflect',
                bias=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size = 3, 
                stride = pool, 
                padding = 1),
            nn.Dropout(.2)
            )
        
    def forward(self, x):
        return(self.cnn(x))
    
        
        
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        example = torch.zeros((1,) + x_shape)
        
        self.cnn = nn.Sequential(
            one_cnn(1, 32, 2),
            one_cnn(32, 32, 2)
            )
        
        example = self.cnn(example).flatten(1)
        quantity = example.shape[-1]
            
        self.lin = nn.Sequential(
            nn.Linear(quantity, 64),
            nn.LeakyReLU(),
            nn.Dropout(.2),
            BayesianLinear(64, 10),
            nn.Softmax(dim = -1)
            )
        
        self.apply(init_weights)
        self.to(device)
        
    def forward(self, x):
        x = x.to(device).float()
        x = (x - 128 * torch.ones(x.shape).to(device)) / 128
        x = self.cnn(x).flatten(1)
        x = self.lin(x)
        return(x)
    
classifier = Classifier()
opti = Adam(classifier.parameters())
