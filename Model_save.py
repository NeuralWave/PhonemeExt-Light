import torch
import math
import torch.nn.functional as funct

import Params

class SingleConvPR(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,5), padding='same', stride=1, dilation=1, bias=True, w_init_gain='leaky_relu'):
        super(SingleConvPR, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class SingleLinearPR(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='leaky_relu'):
        super(SingleLinearPR, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvLayersPR(torch.nn.Module):
    def __init__(self, nConvLayers=10):
        super(ConvLayersPR, self).__init__()
        self.convolutions = torch.nn.ModuleList()

        self.convolutions.append(
            torch.nn.Sequential(
                SingleConvPR(1, 1, kernel_size=(Params.model_params.KERNEL_CONV_1,Params.model_params.KERNEL_CONV_2), stride=1, padding='same', dilation=1, w_init_gain='leaky_relu'),
                torch.nn.MaxPool2d(kernel_size=(Params.model_params.KERNEL_POOL_1,Params.model_params.KERNEL_POOL_2))
                ,torch.nn.BatchNorm2d(1,momentum=Params.model_params.BATCHNORM_MOMENTUM,track_running_stats=Params.model_params.BATCHNORM_RUN_STATS)
            )
        )

        for i in range(1, nConvLayers):
            self.convolutions.append(
                torch.nn.Sequential(
                    SingleConvPR(1, 1, kernel_size=(Params.model_params.KERNEL_CONV_1,Params.model_params.KERNEL_CONV_2), stride=1, padding='same', dilation=1, w_init_gain='leaky_relu')
                    ,torch.nn.BatchNorm2d(1,momentum=Params.model_params.BATCHNORM_MOMENTUM,track_running_stats=Params.model_params.BATCHNORM_RUN_STATS)
                )
            )
        self.dropout = torch.nn.Dropout(p=Params.model_params.DROPOUT_CONV)
    
    def forward(self, x):
        x = funct.leaky_relu(self.convolutions[0](x))
        for i in range(1, len(self.convolutions)):
             x = self.dropout(funct.leaky_relu(self.convolutions[i](x)))

        return x


class LinLayersPR(torch.nn.Module):
    def __init__(self, nLinLayers=3):
        super(LinLayersPR, self).__init__()
        self.linear = torch.nn.ModuleList()

        self.linear.append(
            torch.nn.Sequential(
                SingleLinearPR(in_dim=math.floor(Params.default.NMELCHANNELS/Params.model_params.KERNEL_POOL_1), out_dim=1024))
        )

        for i in range(1, nLinLayers):
            self.linear.append(
                torch.nn.Sequential(
                    SingleLinearPR(in_dim=1024, out_dim=1024))
            )

        self.linear.append(
            torch.nn.Sequential(
                SingleLinearPR(in_dim=1024, out_dim=Params.default.NPHNCLASSES))
        )

        self.dropout = torch.nn.Dropout(p=Params.model_params.DROPOUT_LIN)

    def forward(self, x):
        #x = funct.leaky_relu(self.linear[0](x))
        for i in range(len(self.linear)-1):
            x = self.dropout(funct.leaky_relu(self.linear[i](x)))
        
        x = funct.leaky_relu(self.linear[-1](x))

        return x


class PRnet(torch.nn.Module):
    def __init__(self, nConvLayers=Params.model_params.N_CONV_LAYERS, nLinLayers=Params.model_params.N_LIN_LAYERS):
        super().__init__()
        self.convLayers = ConvLayersPR(nConvLayers)
        self.linLayers = LinLayersPR(nLinLayers)

    def get_n_params(self):
        print(sum(p.numel() for p in self.parameters() if p.requires_grad))
    
    def forward(self, x):
        y_conv = self.convLayers(x)
        y_conv = y_conv.transpose(2,3)
        y_conv = torch.squeeze(y_conv, 1)
        y = self.linLayers(y_conv)  # (N,T,nClasses)
        y = funct.log_softmax(y, dim=2)

        return y