# import paddle
# import numpy as np
#
#
# class qkv_super(paddle.nn.Linear):
#
#     def __init__(self, super_in_dim, super_out_dim, bias=True, uniform_=
#         None, non_linear='linear', scale=False):
#         super().__init__(super_in_dim, super_out_dim)
#         self.super_in_dim = super_in_dim
#         self.super_out_dim = super_out_dim
#         self.sample_in_dim = None
#         self.sample_out_dim = None
#         self.samples = {}
#         self.scale = scale
#         self.profiling = False
#
#     def profile(self, mode=True):
#         self.profiling = mode
#
#     def sample_parameters(self, resample=False):
#         if self.profiling or resample:
#             return self._sample_parameters()
#         return self.samples
#
#     def _reset_parameters(self, bias, uniform_, non_linear):
#         init_XavierUniform = paddle.nn.initializer.XavierUniform()
#         init_XavierUniform(self.weight) if uniform_ is None else uniform_(self
#             .weight, non_linear=non_linear)
#         if bias:
#             init_Constant = paddle.nn.initializer.Constant(value=0.0)
#             init_Constant(self.bias)
#
#     def set_sample_config(self, sample_in_dim, sample_out_dim):
#         self.sample_in_dim = sample_in_dim
#         self.sample_out_dim = sample_out_dim
#         self._sample_parameters()
#
#     def _sample_parameters(self):
#         self.samples['weight'] = sample_weight(self.weight, self.
#             sample_in_dim, self.sample_out_dim)
#         self.samples['bias'] = self.bias
#         self.sample_scale = self.super_out_dim / self.sample_out_dim
#         if self.bias is not None:
#             self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
#         return self.samples
#
#     # def forward(self, x):
#     #     self.sample_parameters()
#     #     return paddle.nn.functional.linear(weight=self.samples['weight'].T,
#     #         bias=self.samples['bias'], x=x) * (self.sample_scale if self.
#     #         scale else 1)
#     def forward(self, x):
#         self.sample_parameters()
#         return paddle.nn.functional.linear(x, self.samples['weight'], self.samples['bias']) * (self.sample_scale if self.scale else 1)
#
#     def calc_sampled_param_num(self):
#         assert 'weight' in self.samples.keys()
#         weight_numel = self.samples['weight'].size
#         if self.samples['bias'] is not None:
#             bias_numel = self.samples['bias'].size
#         else:
#             bias_numel = 0
#         return weight_numel + bias_numel
#
#     def get_complexity(self, sequence_length):
#         total_flops = 0
#         total_flops += sequence_length * np.prod(tuple(self.samples[
#             'weight'].shape))
#         return total_flops
#
#
# def sample_weight(weight, sample_in_dim, sample_out_dim):
#     sample_weight = weight[:, :sample_in_dim]
#     sample_weight = paddle.concat(x=[sample_weight[i:sample_out_dim:3, :] for
#         i in range(3)], axis=0)
#     return sample_weight
#
#
# def sample_bias(bias, sample_out_dim):
#     sample_bias = bias[:sample_out_dim]
#     return sample_bias



import paddle
import numpy as np

class qkv_super(paddle.nn.Linear):

    def __init__(self, super_in_dim, super_out_dim, bias=True, uniform_=None, non_linear='linear', scale=False):
        super().__init__(super_in_dim, super_out_dim)
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim
        self.sample_in_dim = None
        self.sample_out_dim = None
        self.samples = {}
        self.scale = scale
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _reset_parameters(self, bias, uniform_, non_linear):
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.weight) if uniform_ is None else uniform_(self.weight, non_linear=non_linear)
        if bias:
            init_Constant = paddle.nn.initializer.Constant(value=0.0)
            init_Constant(self.bias)

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = sample_weight(self.weight.T, self.sample_in_dim, self.sample_out_dim)
        self.samples['bias'] = self.bias if self.bias is None else sample_bias(self.bias, self.sample_out_dim)
        self.sample_scale = self.super_out_dim / self.sample_out_dim if self.sample_out_dim else 1
        return self.samples

    def forward(self, x):
        self.sample_parameters()
        weight = self.samples['weight'].T
        bias = self.samples['bias']
#        print("weight"+weight)
        # print("qkv x ,weight,bias"+x+""+weight+""+bias)
        return paddle.nn.functional.linear(x, weight, bias) * (self.sample_scale if self.scale else 1)

    # def forward(self, x):
    #     self.sample_parameters()
    #     # 确保 weight 的形状是 [528, 输出特征数量]
    #     # 这里需要根据实际的输出特征数量进行替换
    #     weight = self.samples['weight'].T  # 假设原始 weight 是 [输出特征数量, 528]
    #     bias = self.samples['bias']
    #
    #     # 打印权重的形状，以确保它是正确的
    #     print("weight shape:", weight.shape)
    #
    #     # 调整 x 的形状以匹配权重矩阵的形状
    #     # 将 x 从 [32, 5, 528] 调整为 [160, 528]
    #     x_reshaped = x.reshape([32 * 5, 528])  # 确保 x_reshaped 的形状是 [160, 528]
    #
    #     # 执行线性变换
    #     out = paddle.nn.functional.linear(x_reshaped, weight, bias)
    #
    #     # 如果存在缩放因子，应用它
    #     out = out * (self.sample_scale if self.scale else 1)
    #
    #     return out
    # def forward(self, x):
    #     self.sample_parameters()
    #     weight = self.samples['weight']
    #     bias = self.samples['bias']
    #     assert x.shape[-1] == weight.shape[
    #         1], f"Input dimension {x.shape[-1]} does not match weight dimension {weight.shape[1]}"
    #     return paddle.nn.functional.linear(x, weight, bias) * (self.sample_scale if self.scale else 1)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].size
        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].size
        else:
            bias_numel = 0
        return weight_numel + bias_numel

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length * np.prod(tuple(self.samples['weight'].shape))
        return total_flops

def sample_weight(weight, sample_in_dim, sample_out_dim):
    sampled_weight = weight[:sample_out_dim, :sample_in_dim]
    return sampled_weight

def sample_bias(bias, sample_out_dim):
    sampled_bias = bias[:sample_out_dim]
    return sampled_bias
