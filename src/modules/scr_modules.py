import torch
import torch.nn as nn


class StyleExtractor(nn.Module):

    def __init__(self, encoder):

        super(StyleExtractor, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:6])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[6:13])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[13:20])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[20:33])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[33:46])  # relu4_1 -> relu5_1
        self.enc_6 = nn.Sequential(*enc_layers[46:69])  # relu5_1 -> relu

        self.conv1x1_0 = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=True)
        self.conv1x1_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=True)
        self.conv1x1_2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=True)
        self.conv1x1_3 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=True)
        self.conv1x1_4 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=True)
        self.conv1x1_5 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)
        
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(6):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def forward(self, input, index):

        feats = self.encode_with_intermediate(input)
        codes = []
        for x in index.split(','):
            code = feats[int(x)].clone()
            gap = torch.nn.functional.adaptive_avg_pool2d(code, (1,1))
            gmp = torch.nn.functional.adaptive_max_pool2d(code, (1,1))
            conv1x1 = getattr(self, 'conv1x1_{:d}'.format(int(x)))
            code = torch.cat([gap, gmp], 1)
            code = self.relu(conv1x1(code))
            codes.append(code)
        return codes 


class Projector(nn.Module):
    def __init__(self,):
        super(Projector, self).__init__()
        self.projector0 = nn.Sequential(
            nn.Linear(64, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
        )
        self.projector1 = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
        )
        self.projector2 = nn.Sequential(
            nn.Linear(256,1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
        )
        self.projector3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
        )
        self.projector4 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
        )
        self.projector5 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
        )

    def forward(self, input, index):

        num = 0
        projections = []
        for x in index.split(','):
            projector = getattr(self, 'projector{:d}'.format(int(x)))        
            code = input[num].view(input[num].size(0), -1)
            projection = projector(code).view(code.size(0), -1)
            projection = nn.functional.normalize(projection)
            projections.append(projection)
            num += 1
        return projections


def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


vgg = make_layers([3, 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M', 512, 512, 'M', 512, 512, 'M'])
