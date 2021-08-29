import torch
import torch.nn as nn

class DenseLayer(nn.Module):        #also called bottleneck Layer (Used for DenseNet-BC Variants)
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels * 4, kernel_size = 1, stride = 1, padding = 0, bias = False)
        #1x1 kernel hat ko + k*(l-1) input channels, und erzeugt 4*32 outputs, welche zu 3x3 kernel gehen. Dies dient vor allem der Parameterreduzierung
        self.bn2 = nn.BatchNorm2d(out_channels * 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels * 4, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        
    def forward(self, x):
        out1 = self.conv1(self.relu1(self.bn1(x)))  #input wird zunächst von 1x1 kernel gefiltert
        out = self.conv2(self.relu2(self.bn2(out1))) #anschließend wird 3x3 kernel genutzt, welcher 32 output channels erzeugt
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
                
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.pool = nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
        
    def forward(self, x):
        out = self.pool(self.conv(self.relu(self.bn(x)))) 
        return out
    
class DenseBlock(nn.Module):
    def __init__(self, layer_size, in_channels, growth_rate):   #growth_rate (k in paper) is equal to amount of output channels for each layer
        super(DenseBlock, self).__init__()
        
        self.block = []
        for i in range(layer_size):
            self.block.append(DenseLayer(in_channels+i*growth_rate,growth_rate))      #input = k0 + (l-1)*k, k0 = in_channels, l-1 = i (l = amount of layers before current layer)
            
        self.block = nn.Sequential(*self.block)
        
    def forward(self, x):
        out = self.block(x)
        return out
    
class DenseNet(nn.Module):
    def __init__(self, in_channels=2, layer_size=[6,6,7,8], growth_rate=12, additional_neurons = 8, dropout_prob = 0.5):  #in_channels describes picture channels (3 for rgb)  #layer_size must be list of 4 Elements
        super(DenseNet, self).__init__()
        
        self.out_channels = 2 * growth_rate #for k = 32 ->  64
        self.in_channels = 0    #used later
        
        #First Convolution & Pooling
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=7, stride = 2, padding = 3, bias = False)
        self.batchnorm1 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        #FirstDenseBlock
        self.in_channels = self.out_channels #2*growth_rate -> 64
        self.out_channels = growth_rate
        self.DenseBlock1 = DenseBlock(layer_size[0], self.in_channels, growth_rate) #6 Denselayer
        
        #First Transition Layer
        self.in_channels = self.in_channels + growth_rate * layer_size[0] #256
        self.out_channels = int(self.in_channels/2) #128
        self.TransitionLayer1 = TransitionBlock(self.in_channels, self.out_channels)
        
        #Second DenseBlock
        self.in_channels = self.out_channels    #128 in channels
        self.out_channels = growth_rate
        self.DenseBlock2 = DenseBlock(layer_size[1], self.in_channels, growth_rate) #12 Denselayer         
        
        #Second Transition Layer
        self.in_channels = self.in_channels + growth_rate * layer_size[1] #512
        self.out_channels = int(self.in_channels/2) #256
        self.TransitionLayer2 = TransitionBlock(self.in_channels, self.out_channels)
        
        #Third DenseBlock
        self.in_channels = self.out_channels    #256 in channels
        self.out_channels = growth_rate
        self.DenseBlock3 = DenseBlock(layer_size[2], self.in_channels, growth_rate) #24 Denselayer  
        
        #Third Transition Layer
        self.in_channels = self.in_channels + growth_rate * layer_size[2] #1024
        self.out_channels = int(self.in_channels/2) #512
        self.TransitionLayer3 = TransitionBlock(self.in_channels, self.out_channels)
        
        #Fourth DenseBlock
        self.in_channels = self.out_channels    #512 in channels
        self.out_channels = growth_rate
        self.DenseBlock4 = DenseBlock(layer_size[3], self.in_channels, growth_rate) #16 Denselayer
        
        #Global Average Pooling -> Compresses all channels of size 3x2 to size 1x1 (for input neurons)
        self.global_avg_pool = nn.AvgPool2d(kernel_size=(2,3),stride=4,padding=0)       #kernel size of 3x2 depends on input size of image!!!!
        
        self.in_channels = self.in_channels + growth_rate*layer_size[3] #1024
        self.batchnorm2 = nn.BatchNorm2d(self.in_channels)
        
        #fully connected layer
        self.in_channels = self.in_channels+additional_neurons
        self.fully_connected = nn.Linear(self.in_channels, 4)     #in_features = 1024 + additional neurons used as input, output = 4 (steering_left, steering_right, acceleration, brake)
        
        self.sigmoid = nn.Sigmoid()
        
        #dropout function
        self.dropout = nn.Dropout(dropout_prob)
        
        #initialization of all weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x, a):   #x = Tensor of Image (96*85*in_channels -> specified in constructor of DenseNet), a = List of other Inputs
        x = self.maxpool(self.relu1(self.batchnorm1(self.conv1(x))))
        x = self.TransitionLayer1(self.DenseBlock1(x))       
        x = self.TransitionLayer2(self.DenseBlock2(x))
        x = self.TransitionLayer3(self.DenseBlock3(x))
        x = self.DenseBlock4(x)
        x = self.global_avg_pool(x)
        x = self.batchnorm2(x)
        x = torch.flatten(x, 1)
        if not torch.cuda.is_available():
          a = torch.Tensor(a)
        a = torch.squeeze(a, 0)
        x = torch.squeeze(x, 0)
        if(len(a.size()) > 1):  #checks if a is passed as batch or single value
            x = torch.cat((x,a),1)      #concatenate tensor x and a alsong dimension 1, since dimension zero is reserved by batch
        else:
            x = torch.cat((x,a),0)      #concatenate tensor x and a along dimension 0 (in evaluation mode)
        x = self.dropout(x)         #set some values of tensor x randomly (P = 0.5) to zero, before passing it to the fc layer
        output = self.fully_connected(x)
        output = self.sigmoid(output)
        return output
    

def count_parameters(model):        #Function to count learnable parameters of model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)