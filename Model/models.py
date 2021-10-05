import torch
import torchvision
from .utils import overrides
import torch.nn.functional as F

##all models use same classes
classes = ['child',
           'male',
           'female']

class CNNClassifier(torch.nn.Module):
    def __init__(self, n_classes=len(classes), layers=[32, 64, 128, 256], n_input_channels=3, kernel_size=5):
        super().__init__()
        self.classes = classes
        L = []
        c = n_input_channels
        for l in layers:
            L.append(torch.nn.Conv2d(c, l, kernel_size, stride=2, padding=kernel_size//2))
            L.append(torch.nn.ReLU())
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, n_classes)

    def forward(self, x):
        return self.classifier(self.network(x).mean(dim=[2, 3]))

    def predict(self, image):
        '''takes one image torch tensor outputs class'''
        logits = self.forward(image[None].float())
        argmax = int(torch.argmax(logits))
        # print(logits)
        return self.classes[argmax]





class LastLayer_Alexnet(torch.nn.Module):
    def __init__(self, n_classes=len(classes)):
        super().__init__()
        self.classes = classes
        self.network = torchvision.models.alexnet(pretrained=True) ##first time it will download weights
        self.new_layer = torch.nn.Linear(4096, n_classes)
        self.network.classifier[6] = self.new_layer
        self.features_layer =  torch.nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        self.network.features[0] = self.features_layer
        self.full = torch.nn.Sequential(self.features_layer, self.new_layer) 
    def forward(self, x):
        return self.network(x)

    def predict(self, image):
        '''takes one image torch tensor outputs class'''
        logits = self.forward(image[None].float())
        argmax = int(torch.argmax(logits))
        return self.classes[argmax]

    @overrides(torch.nn.Module)
    def parameters(self, recurse: bool = True):
        return self.full.parameters()


    

class MyResNet(torch.nn.Module):
    def __init__(self, in_channels=1):
        super(MyResNet, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=False)
        self.classes = classes
     # original definition of the first layer on the renset class
     # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    def predict(self, image):
        '''takes one image torch tensor outputs class'''
        logits = self.forward(image[None].float())
        argmax = int(torch.argmax(logits))
        return self.classes[argmax]
    def forward(self, x):
        return self.model(x)



'mid layer stores code'
class FCN_AUTOENCODER(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
            self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            self.b3 = torch.nn.BatchNorm2d(n_output)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))

    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                               stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))

    def __init__(self, layers=[8, 16, 32, 1], n_class=1, kernel_size=5, use_skip=True):
        """
           Your code here.
           Setup your detection network
        """
        super().__init__()
        ##vals for input normalization
        self.input_mean = torch.Tensor([0.5, 0.5, 0.5])
        self.input_std = torch.Tensor([0.5, 0.5, 0.5])

        c = 1
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [1] + layers[:-1]
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, self.Block(c, l, kernel_size, 2))
            c = l
        # Produce lower res output
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]
        self.classifier = torch.nn.Conv2d(c, n_class, 1)
        self.size = torch.nn.Conv2d(c, 2, 1)
        self.mid_layer = None

    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
        """
        ##instance normalization
        # z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
        size = x.size()
        z = x.view(size[0],size[1], 100, 160)
        up_activation = []
        for i in range(self.n_conv):
            # Add all the information required for skip connections
            up_activation.append(z)
            z = self._modules['conv%d' % i](z)

        self.mid_layer = F.sigmoid(z.clone()) ##code

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d' % i](z)
            # Fix the padding
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
            # Add the skip connection
            if self.use_skip:
                z = torch.cat([z, up_activation[i]], dim=1)

        z = self.classifier(z)
        z = z.view(size)
        # mean_c1 = torch.Tensor([[0.1,1,1,1,1,1,0,0,0,0] for i in range(z.size(0))], requires_grad=True)
        output = torch.randn(z.size(0), 10, requires_grad=True)

        return z ##sigmoid output




def save_model(model, info):
    from torch import save
    from os import path
    # if isinstance(model, CNNClassifier):
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'saved_models/'  + info + ".th" ))
    # raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_weights(model, my_path=None):
    from torch import load
    r = model
    if my_path:
        r.load_state_dict(load(my_path, map_location='cpu'))
    return r
