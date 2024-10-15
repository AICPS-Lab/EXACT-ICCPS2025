import math
import warnings
import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np
from models import ASPP, aspp, TemporalPositionalEmbedding
from torch.nn import MultiheadAttention
from torch.nn import LSTM
# torch.__version__ = '2.1.0+cu121'
def seasonality_model(thetas, length=200):
    device = thetas.device
    t = torch.linspace(0, 1, length, device=device)
    p = thetas.size()[-1]
    
    assert p <= thetas.shape[1], 'thetas_dim is too big.'
    
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    
    s1 = torch.stack([torch.cos(2 * np.pi * i * t) for i in range(p1)]).float()
    s2 = torch.stack([torch.sin(2 * np.pi * i * t) for i in range(p2)]).float()
    S = torch.cat([s1, s2])
    
    return thetas.mm(S.T)
class Learner(nn.Module):
    """

    """

    def __init__(self, config):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()


        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList() # nn.ParameterList() acts like register_parameter() but for list of parameters
        # running_mean and running_var
        # self.vars_bn = nn.ParameterList() # remove vars_bn as it is register_buffer() instead of register_parameter()
        # positional encoding list:
        # https://discuss.pytorch.org/t/solved-register-parameter-vs-register-buffer-vs-nn-parameter/31953
        # just call register_buffer?
        # self.vars_pe = nn.ParameterList()        

        for i, (name, param) in enumerate(self.config):
            if name == 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name == 'conv1d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:3]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
            elif name == 'lstm':
                weight_ih_l0 = nn.Parameter(torch.ones(param[1]*4, param[0]))
                weight_hh_l0 = nn.Parameter(torch.ones(param[1]*4, param[1]))
                bias_ih_l0 = nn.Parameter(torch.zeros(param[1]*4))
                bias_hh_l0 = nn.Parameter(torch.zeros(param[1]*4))
                nn.init.kaiming_uniform_(weight_ih_l0)
                nn.init.kaiming_uniform_(weight_hh_l0)

                weight_ih_l1 = nn.Parameter(torch.ones(param[1]*4, param[1]))
                weight_hh_l1 = nn.Parameter(torch.ones(param[1]*4, param[1]))
                bias_ih_l1 = nn.Parameter(torch.zeros(param[1]*4))
                bias_hh_l1 = nn.Parameter(torch.zeros(param[1]*4))
                nn.init.kaiming_uniform_(weight_ih_l1)
                nn.init.kaiming_uniform_(weight_hh_l1)
                
                self.vars.append(weight_ih_l0)
                self.vars.append(weight_hh_l0)
                self.vars.append(bias_ih_l0)
                self.vars.append(bias_hh_l0)
                self.vars.append(weight_ih_l1)
                self.vars.append(weight_hh_l1)
                self.vars.append(bias_ih_l1)
                self.vars.append(bias_hh_l1)

            elif name == 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'attention':
                # multihead attention
                qkv = param[0] *3
                in_proj_weight = nn.Parameter(torch.ones(qkv, param[0]))
                torch.nn.init.kaiming_normal_(in_proj_weight)
                in_proj_bias = nn.Parameter(torch.zeros(qkv))

                out_proj_weight = nn.Parameter(torch.ones(param[0], param[0]))
                torch.nn.init.kaiming_normal_(out_proj_weight)
                out_proj_bias = nn.Parameter(torch.zeros(param[0]))

                self.vars.append(in_proj_weight)
                self.vars.append(in_proj_bias)
                self.vars.append(out_proj_weight)
                self.vars.append(out_proj_bias)
            elif name == "aspp":
                for _ in range(len(param[3])):
                    w = nn.Parameter(torch.ones(param[0:3]))
                    torch.nn.init.kaiming_normal_(w)
                    self.vars.append(w)
                    self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name == "tpe":
                w = nn.Parameter(torch.randn(1, param[0], param[1]) * 0.01)
                # torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)

                pe = torch.zeros(1, param[0], param[1])
                position = torch.arange(0., param[0]).unsqueeze(1)
                div_term = torch.exp(torch.arange(0., param[1], 2) * -(math.log(10000.0) / param[1]))
                pe[..., 0::2] = torch.sin(position * div_term)
                pe[..., 1::2] = torch.cos(position * div_term)
                pe = nn.Parameter(pe, requires_grad=False)
                # self.vars_pe.append(pe)
                self.register_buffer('pe', pe)

            elif name == 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.register_buffer('running_mean' + str(param[0]), running_mean)
                self.register_buffer('running_var' + str(param[0]), running_var)
                # self.vars_bn.extend([running_mean, running_var])

            elif name == 'neural_basis':
                pass
            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid'] + ['upsample1d', 'avg_pool1d', 'max_pool1d'] + \
                        ['adaptive_avg_pool1d', 'aspp', 'indicing', 'upsample_unet', 'store_conv', 'dropout']:
                continue
            else:
                raise NotImplementedError(name + ' not implemented')






    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name == 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'
            elif name == 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'
            elif name == 'conv1d':
                tmp = 'conv1d:(ch_in:%d, ch_out:%d, k:%d, stride:%d, padding:%d)'\
                    %(param[1], param[0], param[2], param[3], param[4])
                info += tmp + '\n'
            elif name == 'lstm':
                tmp = 'lstm:(input_size:%d, hidden_size:%d, num_layer:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name == 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'
            elif name == 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'
            elif name == 'attention':
                # (6, 1) embed_dim, head
                tmp = 'attention:(embed_dim:%d, head:%d)'%(param[0], param[1])
                info += tmp + '\n'
            elif name == 'aspp':
                tmp = 'aspp:(in:%d, out:%d, k:%d, num_dilations=padding:%s)'%(param[0], param[1], param[2], param[3])
                info += tmp + '\n'
            elif name == 'tpe':
                tmp = 'tpe:(time_steps:%d, d_model:%d)'%(param[0], param[1])
                info += tmp + '\n'
            elif name == 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name == 'avg_pool1d':
                tmp = 'avg_pool1d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name == 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name == 'max_pool1d':
                tmp = 'max_pool1d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name == 'adaptive_avg_pool1d':
                tmp = 'adaptive_avg_pool1d:(output_size:%d)'%(param[0])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn', 
                          'upsample_unet', 'indicing', 'store_conv', 'dropout', 'neural_basis']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info



    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """
        x = x.transpose(1, 2)

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0
        # pe = 0

        store_conv = []

        for name, param in self.config:
            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'conv1d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                # print('forward:', idx, x.norm().item())
                x = F.conv1d(x, w, b, stride=param[3], padding=param[4])
                #NOTE: store the last conv layer for unet
                idx += 2
            elif name == 'store_conv':
                store_conv.append(x)
            elif name == 'lstm':
                x = x.transpose(1, 2)
                w_ih_l0, w_hh_l0, b_ih_l0, b_hh_l0 = vars[idx], vars[idx+1], vars[idx+2], vars[idx+3]
                w_ih_l1, w_hh_l1, b_ih_l1, b_hh_l1 = vars[idx+4], vars[idx+5], vars[idx+6], vars[idx+7]
                state = {'weight_ih_l0': w_ih_l0, 'weight_hh_l0': w_hh_l0, 'bias_ih_l0': b_ih_l0, 'bias_hh_l0': b_hh_l0,
                         'weight_ih_l1': w_ih_l1, 'weight_hh_l1': w_hh_l1, 'bias_ih_l1': b_ih_l1, 'bias_hh_l1': b_hh_l1}
                functional_out = torch.func.functional_call(LSTM(param[0], param[1], param[2], batch_first=True), state, (x))
                x = functional_out[0]
                x = x.transpose(1, 2)
                idx += 8
            elif name == 'tpe':
                tw = vars[idx]
                pw = dict(self.named_buffers())['pe'] #.expand(tw.shape[0], -1, -1)
                # pw = self.vars_pe[pe]
                # pe += 1
                state = {'temporal_embeddings': tw, 
                         'positional_embeddings': pw}
                x = torch.func.functional_call(TemporalPositionalEmbedding(param[0], param[1]), state, (x))
                idx += 1
            elif name == 'neural_basis':
                # print('forward:', idx, x.norm().item())
                x = seasonality_model(x, length=param[0])
                # print(name, param, '\tout:', x.shape)
            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                # load from running_mean, running_var register_buffer:
                running_mean = dict(self.named_buffers())['running_mean' + str(param[0])]
                running_var = dict(self.named_buffers())['running_var'+ str(param[0])]
                # running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                # bn_idx += 2
            elif name == 'attention':
                # print('forward:', idx, x.norm().item())
                x = x.transpose(1, 2)
                qkv, qkv_b, w1, w1_b = vars[idx], vars[idx+1], vars[idx+2], vars[idx+3]
                x = F.multi_head_attention_forward(x, x, x, param[0], param[1], in_proj_weight=qkv, in_proj_bias=qkv_b, 
                               bias_k=None, bias_v=None, add_zero_attn=False,
                               out_proj_weight=w1, out_proj_bias=w1_b, dropout_p=0.0)[0]
                x = x.transpose(1, 2)
                idx += 4
            elif name == 'aspp':
                # print('forward:', idx, x.norm().item())
                Ws = []
                Bs = []
                for _ in range(len(param[3])):
                    Ws.append(vars[idx])
                    Bs.append(vars[idx+1])
                    idx += 2
                x = aspp(x, Ws, Bs, param[3])
            elif name == "adaptive_avg_pool1d":
                x = F.adaptive_avg_pool1d(x, param[0])
            elif name == 'flatten':
                # print('flatten', x.shape)
                x = x.view(x.size(0), -1)
            elif name == 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name == 'tanh':
                x = F.tanh(x)
            elif name == 'sigmoid':
                x = torch.sigmoid(x)
            elif name == 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name == 'upsample_unet':
                #NOTE: this  only for unet, meaning it will get the last conv layer and concat it with the upsampled layer
                x = F.upsample_nearest(x, scale_factor=param[0])
                x = torch.cat([x, store_conv.pop()], dim=1)
            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name == 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            elif name == 'max_pool1d':
                x = F.max_pool1d(x, param[0], param[1], param[2])
            elif name == 'avg_pool1d':
                x = F.avg_pool1d(x, param[0], param[1], param[2])
            elif name == 'dropout':
                x = F.dropout(x, p=param[0], training=param[1])
            elif name == 'indicing':
                # last 50 indice for example from param 0
                x = x.squeeze(1)
                assert param[0] <= x.shape[1]
                x = x[:,  -param[0]:]
            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        # assert bn_idx == len(self.vars_bn)

        return x


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars
    


    # def parameters(self):
    #     # https://discuss.pytorch.org/t/overriding-module-parameters-with-dataparallel/25217/5
    #     for param in self.vars:
    #         # if only_trainable and not param.requires_grad:
    #         #     continue
    #         yield param