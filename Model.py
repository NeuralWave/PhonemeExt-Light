import torch
import math
import torch.nn.functional as funct

import Params

class SingleConvPR(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,5), padding='same', stride=1, dilation=1, bias=False, w_init_gain='leaky_relu'):
        super(SingleConvPR, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias, groups=in_channels)

        # torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
        torch.nn.init.uniform_(self.conv.weight, a=-0.05, b=0.05)

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class SingleLinearPR(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='leaky_relu'):
        super(SingleLinearPR, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        # torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
        torch.nn.init.uniform_(self.linear_layer.weight, a=-0.05, b=0.05)
        torch.nn.init.uniform_(self.linear_layer.bias, a=-0.05, b=0.05)

    def forward(self, x):
        return self.linear_layer(x)


class ConvLayersPR(torch.nn.Module):
    def __init__(self, nConvLayers_1=4, nConvLayers_2=6, nFeatures_1=128, nFeatures_2=256):
        super(ConvLayersPR, self).__init__()
        self.convolutions = torch.nn.ModuleList()

        self.convolutions.append(
            torch.nn.Sequential(
                SingleConvPR(1, nFeatures_1, kernel_size=(Params.model_params.KERNEL_CONV_1,Params.model_params.KERNEL_CONV_2), stride=1, padding='same', dilation=1, w_init_gain='leaky_relu'),
                torch.nn.MaxPool2d(kernel_size=(Params.model_params.KERNEL_POOL_1,Params.model_params.KERNEL_POOL_2))
                ,torch.nn.BatchNorm2d(nFeatures_1,momentum=Params.model_params.BATCHNORM_MOMENTUM,track_running_stats=Params.model_params.BATCHNORM_RUN_STATS)
            )
        )

        for i in range(1, nConvLayers_1):
            self.convolutions.append(
                torch.nn.Sequential(
                    SingleConvPR(nFeatures_1, nFeatures_1, kernel_size=(Params.model_params.KERNEL_CONV_1,Params.model_params.KERNEL_CONV_2), stride=1, padding='same', dilation=1, w_init_gain='leaky_relu')
                    ,torch.nn.BatchNorm2d(nFeatures_1,momentum=Params.model_params.BATCHNORM_MOMENTUM,track_running_stats=Params.model_params.BATCHNORM_RUN_STATS)
                )
            )

        self.convolutions.append(
            torch.nn.Sequential(
                SingleConvPR(nFeatures_1, nFeatures_2, kernel_size=(Params.model_params.KERNEL_CONV_1,Params.model_params.KERNEL_CONV_2), stride=1, padding='same', dilation=1, w_init_gain='leaky_relu')
                ,torch.nn.BatchNorm2d(nFeatures_2,momentum=Params.model_params.BATCHNORM_MOMENTUM,track_running_stats=Params.model_params.BATCHNORM_RUN_STATS)
            )
        )


        for i in range(1, nConvLayers_2):
            self.convolutions.append(
                torch.nn.Sequential(
                    SingleConvPR(nFeatures_2, nFeatures_2, kernel_size=(Params.model_params.KERNEL_CONV_1,Params.model_params.KERNEL_CONV_2), stride=1, padding='same', dilation=1, w_init_gain='leaky_relu')
                    ,torch.nn.BatchNorm2d(nFeatures_2,momentum=Params.model_params.BATCHNORM_MOMENTUM,track_running_stats=Params.model_params.BATCHNORM_RUN_STATS)
                )
            )

        self.dropout = torch.nn.Dropout(p=Params.model_params.DROPOUT_CONV)
    
    def forward(self, x):
        x = funct.leaky_relu(self.convolutions[0](x))
        # x , j = torch.max(x, dim=1, keepdim=True)
        
        for i in range(1, len(self.convolutions)):
             x = funct.leaky_relu(self.convolutions[i](x))
             # x , j = torch.max(x, dim=1, keepdim=True)
             x = self.dropout(x)

        return x



class LinLayersPR(torch.nn.Module):
    def __init__(self, nLinLayers=3, linDim=1024):
        super(LinLayersPR, self).__init__()
        self.linear = torch.nn.ModuleList()

        self.linear.append(
            torch.nn.Sequential(
                SingleLinearPR(in_dim=Params.model_params.CONV_FEAT_OUT*math.floor(Params.default.NMELCHANNELS/Params.model_params.KERNEL_POOL_1), out_dim=linDim))
        )

        for i in range(1, nLinLayers):
            self.linear.append(
                torch.nn.Sequential(
                    SingleLinearPR(in_dim=linDim, out_dim=linDim))
            )

        self.linear.append(
            torch.nn.Sequential(
                SingleLinearPR(in_dim=linDim, out_dim=Params.default.NPHNCLASSES))
        )

        self.dropout = torch.nn.Dropout(p=Params.model_params.DROPOUT_LIN)

    def forward(self, x):
        #x = funct.leaky_relu(self.linear[0](x))
        for i in range(len(self.linear)-1):
            x = self.dropout(funct.leaky_relu(self.linear[i](x)))
        
        x = funct.leaky_relu(self.linear[-1](x))

        return x


class PRnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convLayers = ConvLayersPR(nConvLayers_1=Params.model_params.N_CONV_LAYERS_1,nConvLayers_2=Params.model_params.N_CONV_LAYERS_2,nFeatures_1=Params.model_params.CONV_FEAT_1,nFeatures_2=Params.model_params.CONV_FEAT_2)
        self.linLayers = LinLayersPR(Params.model_params.N_LIN_LAYERS, Params.model_params.LIN_DIM)

    def get_n_params(self):
        print("Num of trainable params =", sum(p.numel() for p in self.parameters() if p.requires_grad))
    
    def forward(self, x):
        y_conv = self.convLayers(x)
        y_conv = y_conv.view(y_conv.size(0), -1, y_conv.size(3))
        y_conv = y_conv.transpose(1,2)
        y = self.linLayers(y_conv)  # (N,T,nClasses)
        y = funct.log_softmax(y, dim=2)

        return y