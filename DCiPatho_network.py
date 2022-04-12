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
        for i in range(num_cross_layers):
            weight_w.append(nn.Parameter(torch.nn.init.normal_(torch.empty(input_dim))))
            weight_b.append(nn.Parameter(torch.nn.init.normal_(torch.empty(input_dim))))
            batchnorm.append(nn.BatchNorm1d(input_dim, affine=False))

        self.weight_w = nn.ParameterList(weight_w)
        self.weight_b = nn.ParameterList(weight_b)
        self.bn = nn.ModuleList(batchnorm)

    def forward(self, x):
        out = x
        x = x.reshape(x.shape[0], -1, 1)
        for i in range(self.num_cross_layers):
            xxTw = torch.matmul(x, torch.matmul(torch.transpose(out.reshape(out.shape[0], -1, 1), 1, 2),
                                                self.weight_w[i].reshape(1, -1, 1)))
            xxTw = xxTw.reshape(xxTw.shape[0], -1)
            out = xxTw + self.weight_b[i] + out

            out = self.bn[i](out)
        return out


class DCiPatho(BaseModel):
    def __init__(self, dense_features_cols):
        super(DCiPatho, self).__init__(config)
        self._num_of_dense_feature = dense_features_cols.__len__()
        self._input_dim = self._num_of_dense_feature
        self.out_layer1 = nn.Linear(self._input_dim, config.out_layer_dims)
        self.out_layer2 = nn.Linear(config.deep_layers[-1], config.out_layer_dims)
        self.residual_layers = nn.ModuleList([
            # 根据稀疏特征的个数创建对应个数的Embedding层，Embedding输入大小是稀疏特征的类别总数，输出稠密向量的维度由config文件配置
            ResidualBlock(self._input_dim, layer) for layer in config.hidden_layers
        ])

        self._deepNet = Deep(self._input_dim, config.deep_layers)
        self._crossNet = Cross(self._input_dim, config.num_cross_layers)

        self._final_dim = 3 * config.out_layer_dims
        self.end_layer1 = nn.Linear(self._final_dim, config.end_dims[0])
        self.end_layer2 = nn.Linear(config.end_dims[0], config.end_dims[1])
        self.end_layer3 = nn.Linear(config.end_dims[1], config.end_dims[2])
        self.end_layer4 = nn.Linear(config.end_dims[2], config.end_dims[3])
        self._final_linear = nn.Linear(config.end_dims[-1], 1)

    def forward(self, x):
        dense_input = x[:, :self._num_of_dense_feature]
        input = dense_input
        for residual in self.residual_layers:
            input = residual(input)
        res_out = torch.relu(self.out_layer1(input))
        # res_out = torch.relu(input)
        # res_shape = res_out.shape
        # res_out = res_out
        deep_out = self._deepNet(dense_input)
        deep_out = torch.relu(self.out_layer2(deep_out))
        cross_out = self._crossNet(dense_input)
        cross_out = torch.relu(self.out_layer1(cross_out))
        # final_input = torch.cat([res_out, deep_out, cross_out], dim=1)
        # final_input = torch.cat([res_out, deep_out, cross_out], dim=1)
        final_input = torch.cat([res_out, deep_out, cross_out], dim=1)
        end1 = self.end_layer1(final_input)
        end2 = self.end_layer2(end1)
        end3 = self.end_layer3(end2)
        end4 = self.end_layer4(end3)
        # output = torch.sigmoid(self._final_linear(final_input))
        output = torch.sigmoid(self._final_linear(end4))
        # try
        # output = torch.relu(self._final_linear(final_input))
        # end_output = torch.sigmoid(output)
        return output
