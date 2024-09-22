import paddle
import numpy as np


class LinearSuper(paddle.nn.Linear):

    def __init__(self, super_in_dim, super_out_dim, bias=True, uniform_=
        None, non_linear='linear', scale=False):
        super().__init__(super_in_dim, super_out_dim)
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim
        self.sample_in_dim = None
        self.sample_out_dim = None
        self.samples = {}
        self.scale = scale
        self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _reset_parameters(self, bias, uniform_, non_linear):
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.weight) if uniform_ is None else uniform_(self
            .weight, non_linear=non_linear)
        if bias:
            init_Constant = paddle.nn.initializer.Constant(value=0.0)
            init_Constant(self.bias)

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        self._sample_parameters()

    # def _sample_parameters(self):
    #     self.samples['weight'] = sample_weight(self.weight, self.
    #         sample_in_dim, self.sample_out_dim)
    #     self.samples['bias'] = self.bias
    #     self.sample_scale = self.super_out_dim / self.sample_out_dim
    #     if self.bias is not None:
    #         self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
    #     return self.samples
    ###
    def _sample_parameters(self):
        self.samples['weight'] = sample_weight(self.weight.T, self.
            sample_in_dim, self.sample_out_dim)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim / self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
        return self.samples
    ###
    def forward(self, x):
        self.sample_parameters()
        return paddle.nn.functional.linear(weight=self.samples['weight'].T,
            bias=self.samples['bias'], x=x) * (self.sample_scale if self.
            scale else 1)

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
        total_flops += sequence_length * np.prod(tuple(self.samples[
            'weight'].shape))
        return total_flops


def sample_weight(weight, sample_in_dim, sample_out_dim):
    sample_weight = weight[:, :sample_in_dim]
    sample_weight = sample_weight[:sample_out_dim, :]
    return sample_weight


def sample_bias(bias, sample_out_dim):
    sample_bias = bias[:sample_out_dim]
    return sample_bias
