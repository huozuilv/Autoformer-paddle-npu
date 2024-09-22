import sys
sys.path.append('E:\\summerwork\\PaConvert-master/paddle_project/utils')
import math
import warnings
from collections.abc import Iterable
from itertools import repeat

import paddle


def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.'
            , stacklevel=2)
    with paddle.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(min=2 * l - 1, max=2 * u - 1)
        tensor.erfinv_()
        tensor.multiply_(y=paddle.to_tensor(std * math.sqrt(2.0)))
        tensor.add_(y=paddle.to_tensor(mean))
        tensor.clip_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = paddle.empty(3, 5)
        >>> paddle.nn.initializer.TruncatedNormal(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# def _ntuple(n):
#
#     def parse(x):
#         if isinstance(x, torch._six.container_abcs.Iterable):
#             return x
#         return tuple(repeat(x, n))
#     return parse
#修改后代码
def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

def drop_path(x, drop_prob: float=0.0, training: bool=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (tuple(x.shape)[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape=shape, dtype=x.dtype)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(paddle.nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
