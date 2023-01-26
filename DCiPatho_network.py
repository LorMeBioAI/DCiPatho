import gc

import numpy as np
import torch
import torch.nn as nn
from BaseModel.basemodel import BaseModel
from DCiPatho_config import Config

config = Config()


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=input_dim, bias=True)

    def forward(self, x):
        out = self.linear2(torch.relu(self.linear1(x)))
        out += x
        out = torch.relu(out)
        return out


class Deep(nn.Module):
    def __init__(self, input_dim, deep_layers):
        super(Deep, self).__init__()

        deep_layers.insert(0, input_dim)
        deep_ayer_list = []
        for layer in list(zip(deep_layers[:-1], deep_layers[1:])):
            deep_ayer_list.append(nn.Linear(layer[0], layer[1]))
            deep_ayer_list.append(nn.BatchNorm1d(layer[1], affine=False))
            deep_ayer_list.append(nn.ReLU(inplace=True))
        self._deep = nn.Sequential(*deep_ayer_list)

    def forward(self, x):
        out = self._deep(x)
        return out


class Cross(nn.Module):
    """
    the operation is this module is x_0 * x_l^T * w_l + x_l + b_l for each layer, and x_0 is the init input
    """

    def __init__(self, input_dim, num_cross_layers):
        super(Cross, self).__init__()

        self.num_cross_layers = num_cross_layers
        weight_w = []
        weight_b = []
        batchnorm = []
        self.para = torch.nn.Parameter(torch.ones(1).requires_grad_())
        self.para.data.fill_(0.5)
        self.x_layer = nn.Linear(10952, 10952)

        for i in range(num_cross_layers):
            weight_w.append(nn.Parameter(torch.nn.init.normal_(torch.empty(input_dim))))
            weight_b.append(nn.Parameter(torch.nn.init.normal_(torch.empty(input_dim))))
            batchnorm.append(nn.BatchNorm1d(input_dim, affine=False))

        self.weight_w = nn.ParameterList(weight_w)
        # print('weight_w', weight_w)
        self.weight_b = nn.ParameterList(weight_b)
        self.bn = nn.ModuleList(batchnorm)

    def forward(self, x):
        out = x
        x = x.reshape(x.shape[0], -1, 1)
        w_list = []
        for i in range(self.num_cross_layers):
            xxTw = torch.matmul(x, torch.matmul(torch.transpose(out.reshape(out.shape[0], -1, 1), 1, 2),
                                                self.weight_w[i].reshape(1, -1, 1)))
            xxTw = xxTw.reshape(xxTw.shape[0], -1)
            # out = self.x_layer(out)
            out = torch.mul(out, self.para)
            out = xxTw + self.weight_b[i] + out
            out = self.bn[i](out)
            w_list.append(self.weight_b[i].detach().cpu().numpy())
        # np.save('output/w_list.npy', np.array(w_list))
        return out


class DCiPatho(BaseModel):
    def __init__(self):
        super(DCiPatho, self).__init__(config)
        self._input_dim = config.freqs_nums
        # self._input_dim = config.freqs_nums - config.nums_dic[7]
        self.out_layer1 = nn.Linear(self._input_dim, config.out_layer_dims)
        self.out_layer2 = nn.Linear(config.deep_layers[-1], config.out_layer_dims)
        self.drop_layer = nn.Dropout(p=config.Dropout)
        self.residual_layers = nn.ModuleList([
            # 根据稀疏特征的个数创建对应个数的Embedding层，Embedding输入大小是稀疏特征的类别总数，输出稠密向量的维度由config文件配置
            ResidualBlock(self._input_dim, layer) for layer in config.hidden_layers
        ])
        self._deepNet = Deep(self._input_dim, config.deep_layers)
        self._crossNet = Cross(self._input_dim, config.num_cross_layers)
        self._final_dim = 3 * config.out_layer_dims
        self.end_layer1 = nn.Linear(self._final_dim, config.end_dims[-3])
        self.end_layer2 = nn.Linear(config.end_dims[-3], config.end_dims[-1])
        # self.end_layer3 = nn.Linear(config.end_dims[-4], config.end_dims[-3])
        # self.end_layer4 = nn.Linear(config.end_dims[-3], config.end_dims[-2])
        self._final_linear = nn.Linear(config.end_dims[-1], 1)

    def forward(self, x):
        input = x
        for residual in self.residual_layers:
            input = residual(input)
        res_out = torch.relu(self.out_layer1(input))
        # res_shape = res_out.shape

        deep_out = self._deepNet(x)
        deep_out = torch.relu(self.out_layer2(deep_out))
        cross_out = self._crossNet(x)
        cross_out = torch.relu(self.out_layer1(cross_out))
        # c_mask = torch.zeros(cross_out.shape, dtype=torch.float)
        # r_mask = torch.zeros(res_out.shape, dtype=torch.float)
        # d_mask = torch.zeros(deep_out.shape, dtype=torch.float)
        # print('cmask', c_mask)
        # c_mask = c_mask.to(device='cuda:0')
        # r_mask = r_mask.to(device='cuda:0')
        # d_mask = d_mask.to(device='cuda:0')
        # final_input = torch.cat([deep_out, cross_out], dim=1)
        # final_input = cross_out
        # final_input = torch.cat([cross_out], dim=1)
        # final_input = torch.cat([cross_out], dim=1)
        final_input = torch.cat([res_out, deep_out, cross_out], dim=1)
        # final_input = torch.cat([deep_out, cross_out, r_mask], dim=1)
        # add Relu
        # final_input = nn.ReLU(self.drop_layer(final_input))
        final_input = self.drop_layer(final_input)

        end1 = self.end_layer1(final_input)
        end2 = self.end_layer2(end1)
        # end3 = self.end_layer3(end2)
        # end4 = self.end_layer4(end3)
        # output = torch.sigmoid(self._final_linear(final_input))
        output = torch.sigmoid(self._final_linear(end2))
        # try
        # output = torch.relu(self._final_linear(final_input))
        # end_output = torch.sigmoid(output)
        return output

    # def reset_weights(m):
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #         m.reset_parameters()
