import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
from utilities.networks_dcnn import *
from utilities.gen_net import *

class simpleNet(nn.Module):
    def __init__(self,Y=True, inp_channels=None):
        super(simpleNet, self).__init__()
        d = 1
        if Y == False:
            d = 3
        if inp_channels is not None:
            self.input = nn.Conv2d(in_channels=inp_channels, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.input = nn.Conv2d(in_channels=d, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)

    
        self.output = nn.Conv2d(in_channels=128, out_channels=d, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        residual = x
        inputs = self.input(self.relu(x))
        out = inputs
        
        out = self.conv1(self.relu(out))
        out = self.conv2(self.relu(out))
        out = self.conv3(self.relu(out))
        out = self.conv4(self.relu(out))
        out = self.conv5(self.relu(out))
        out = self.conv6(self.relu(out))

        #out = torch.add(out, inputs)

        out = self.output(self.relu(out))
        
        out = torch.add(out, residual[:, :3])
        return out

class simpleResNet(nn.Module):
    def __init__(self,Y=True, inp_channels=None):
        super(simpleResNet, self).__init__()
        d = 1
        if Y == False:
            d = 3
        if inp_channels is not None:
            self.input = nn.Conv2d(in_channels=inp_channels, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.input = nn.Conv2d(in_channels=d, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)

    
        self.output = nn.Conv2d(in_channels=128, out_channels=d, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        residual = x
        inputs = self.input(self.relu(x))
        out = inputs
        
        out1 = self.conv1(self.relu(inputs))
        out2 = self.conv2(self.relu(out1) + out)
        out3 = self.conv3(self.relu(out2))
        out4 = self.conv4(self.relu(out3) + out1)
        out5 = self.conv5(self.relu(out4))
        out6 = self.conv6(self.relu(out5) + out3)

        #out = torch.add(out, inputs)

        out = self.output(self.relu(out6))
        
        out = torch.add(out, residual[:, :3])
        return out

class simpleResNetBN(nn.Module):
    def __init__(self,Y=True, inp_channels=None):
        super(simpleResNetBN, self).__init__()
        d = 1
        if Y == False:
            d = 3
        if inp_channels is not None:
            self.input = nn.Sequential(
        nn.Conv2d(in_channels=inp_channels, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        # nn.BatchNorm2d(128),
        nn.Dropout2d(p=0.1),
      )
        else:
            self.input = nn.Sequential(
        nn.Conv2d(in_channels=d, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        # nn.BatchNorm2d(128),
        nn.Dropout2d(p=0.1),
      )
        self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        # nn.BatchNorm2d(128),
        nn.Dropout2d(p=0.1),
    )
        self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        # nn.BatchNorm2d(128),
        # nn.Dropout2d(p=0.1),
    )
        self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        # nn.BatchNorm2d(128),
        # nn.Dropout2d(p=0.1),
    )
        self.conv4 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        # nn.BatchNorm2d(128),
        # nn.Dropout2d(p=0.1),
    )
        self.conv5 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        # nn.BatchNorm2d(128),
        # nn.Dropout2d(p=0.1),
    )
        
        self.conv6 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        # nn.BatchNorm2d(128),
    )

    
        self.output = nn.Sequential(
          nn.Conv2d(in_channels=128, out_channels=d, kernel_size=1, stride=1, padding=0, bias=False),
          # nn.BatchNorm2d(128),
          # nn.ReLU(inplace=False),
          # nn.Conv2d(in_channels=128, out_channels=d, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.relu = nn.ReLU(inplace=False)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        residual = x
        inputs = self.input(self.relu(x))
        out = inputs
        
        out1 = self.conv1(self.relu(inputs))
        out2 = self.conv2(self.relu(out1))
        out3 = self.conv3(self.relu(out2))
        out4 = self.conv4(self.relu(out3))
        out5 = self.conv5(self.relu(out4))
        out6 = self.conv6(self.relu(out5))

        #out = torch.add(out, inputs)
        # out6 = torch.cat([out6, residual], dim=1)

        out = self.output(out6)
        
        out = torch.add(torch.tanh(out), residual[:, :3])
        return out

class ResBuild(nn.Module):

    def __init__(self, in_channels, out_channels): # feel free to modify input paramters
        super(ResBuild, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=256, kernel_size=5, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256+3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
        )
        resNet = models.resnet18(pretrained=True)
        # modules = list(resNet.children())[:-3]
        # resNet = models.segmentation.fcn_resnet50(pretrained=False)
        # resNet = models.segmentation.deeplabv3_resnet50(pretrained=True)
        # resNet = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # print(resNet)
        
        # resNet = torch.hub.load('pytorch/vision:v0.6.0', 'fcn_resnet101', pretrained=True)

        # modules=list(list(resNet.children())[0].children()) + list(list(resNet.children())[1].children()) #+ list(list(resNet.children())[2].children())
        # print("Hello", list(list(resNet.children())[0].children()))
        # sys.exit()
        modules=list(resNet.children())[:-3]
        resNet=nn.Sequential(*modules)
        # for p in resNet.parameters():
        #     p.requires_grad = False
        self.resnet = resNet
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def forward(self, x): # feel free to modify input paramters
        # print("Before Normalization",x.size())
        self.eval()
        img = self.normalize(x[0, :3]).unsqueeze(0)
        shape = img.size()
        # print("After Normalization", img.size())
        # print(img)
        self.resnet.eval()
        h = self.resnet(img)
        h = nn.Upsample(size=(shape[2] ,shape[3]), mode='bilinear')(h)
        stack = torch.cat([h, x], dim=1)
        # print(h)
        # print(x.size(), h.size())
        # sys.exit()
        # h = self.conv1(h)
        h = self.conv3(stack)
        # print("Prediction", h.size()) 
        
        # print(h.size()),
        # return h[:, :, :shape[2], :shape[3]] + x[:, :3]
        # h = nn.Upsample(size=(shape[2] ,shape[3]), mode='bilinear')(h)
        # print(torch.max(torch.abs(h)).item())
        # print(h[0, :, 0, 0])
        return h + x[:, :3]

class DenoiseSuperRes(nn.Module):

    def __init__(self, in_channels, out_channels): # feel free to modify input paramters
        super(DenoiseSuperRes, self).__init__()
        # resNet = CompletionNetwork()
        # resNet.load_state_dict(torch.load("models/model_cn"))
        resNet = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R')
        mod = torch.load("models/dncnn_color_blind.pth")
        resNet.load_state_dict(mod, strict=False)

        self.conv1 = nn.Conv2d(in_channels=64+3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(in_channels=64+3, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout2d(p=0.2),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout2d(p=0.3),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        # )
        # print(resNet)
        for p in resNet.parameters():
            p.requires_grad = False
        self.resnet = resNet
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x): # feel free to modify input paramters
        # print("Before Normalization",x.size())
        # img = self.normalize(x[0, :3]).unsqueeze(0)
        shape = x.size()
        # print("After Normalization", img.size())
        # print(img)
        self.resnet.eval()
        # Completion Net
        # mask = torch.zeros((*shape)).to(x.device)
        # img = torch.cat([x, mask[:,0:1]], dim=1)
        # h = self.resnet(img)

        # Denoise
        h = self.resnet(x)
        # print(h.size()),
        # sys.exit()
        h = nn.Upsample(size=(shape[2] ,shape[3]), mode='bilinear')(h)
        # h = h[:, :, :shape[2], :shape[3]]
        # print(h)
        stack = torch.cat([h, x], dim=1)
        out1 = self.relu(self.conv1(stack))
        out2 = self.relu(self.conv2(out1))
        out3 = self.relu(self.conv3(out2))
        out4 = self.relu(self.conv4(out3))
        out5 = self.relu(self.conv5(out4))
        h = out6 = self.conv6(out5)
        # h = self.conv1(h)
        # h = self.conv2(h)
        # print("Prediction", h.size()) 
        
        # print(h.size()),
        # return h[:, :, :shape[2], :shape[3]] + x[:, :3]
        # h = nn.Upsample(size=(shape[2] ,shape[3]), mode='bilinear')(h)
        # print(torch.max(torch.abs(h)).item())
        # print(h[0, :, 0, 0])
        return h + x[:, :3]








