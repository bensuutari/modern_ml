import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict,List
import torch.nn.functional as F

def calculate_maxpooloutput(hin,win,dilation,padding,kernel_size,stride):
    hout = ((hin + 2*padding[0] - dilation[0]*(kernel_size[0] - 1) - 1)/stride[0])#+1
    wout = ((win + 2*padding[1] - dilation[1]*(kernel_size[1] - 1) - 1)/stride[1])#+1
    return hout,wout


@dataclass
class BasicCNNConfig:
    in_channels: int
    image_height: int
    image_width: int
    num_outputs: int
    classification: bool
    convs: List[Dict] = field(default_factory=lambda:[
            {
                'out_channels' :64, 
                'kernel_size' : (3,3), 
                'stride':(1,1), 
                'padding':(0,0), 
                'dilation':(1,1)
            }
        ]
    )
    max_pools: List[Dict] = field(default_factory=lambda:[
            {
                'kernel_size' : (2,2), 
                'stride':(1,1), 
                'padding':(0,0), 
                'dilation':(1,1),
            }
        ]
    )
    mlp_layers: List[Dict] = field(default_factory=lambda:[
            {
                'size' : 256, 
                'dropout':0.2,
            },
            {
                'size' : 128, 
                'dropout':0.2,
            },
            {
                'size' : 64, 
                'dropout':0.2,
            }
        ]
    )
    def __post_init__(self):
        assert len(self.convs)==len(self.max_pools)



class BasicCNN(nn.Module):
    def __init__(self,config: BasicCNNConfig):
        super(BasicCNN,self).__init__()
        self.classification = config.classification
        conv = config.convs[0]
        conv['in_channels'] = config.in_channels
        mp = config.max_pools[0]
        conv_list = [
            nn.Conv2d(**conv),
            nn.MaxPool2d(**mp),
        ]
        hout,wout = calculate_maxpooloutput(
            config.image_height,
            config.image_width,
            conv['dilation'],
            conv['padding'],
            conv['kernel_size'],
            conv['stride']
        )
        prev_out_channels = conv['out_channels']
        if len(config.convs)>1:
            for conv,mp in zip(config.convs[1:],config.max_pools[1:]):
                conv['in_channels'] = prev_out_channels
                conv_list.append(nn.Conv2d(**conv))
                conv_list.append(nn.MaxPool2d(**mp))
                hout,wout = calculate_maxpooloutput(
                    hout,
                    wout,
                    conv['dilation'],
                    conv['padding'],
                    conv['kernel_size'],
                    conv['stride']
                )
                prev_out_channels = conv['out_channels']
        self.convolution = nn.Sequential(*conv_list)
        mlp_list = [
            nn.Linear(int(hout*wout*prev_out_channels),config.mlp_layers[0]['size']),
            nn.ReLU(),
            nn.Dropout(config.mlp_layers[0]['dropout'])
        ]
        
        if len(config.mlp_layers)>1:
            last_out = config.mlp_layers[0]['size']
            for l in config.mlp_layers[1:]:
                mlp_list.append(nn.Linear(last_out,l['size']))
                mlp_list.append(nn.ReLU())
                mlp_list.append(nn.Dropout(l['dropout']))
                last_out = l['size']
        mlp_list.append(nn.Linear(l['size'],config.num_outputs))
        self.mlp = nn.Sequential(*mlp_list)
    def forward(self,x):
        x = self.convolution(x)
        x = x.flatten(-2,-1).flatten(-2)
        if self.classification:
            return F.softmax(self.mlp(x))
        else:
            return self.mlp(x)

