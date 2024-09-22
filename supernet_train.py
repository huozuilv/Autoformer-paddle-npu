import argparse
import datetime
import numpy as np
import time
import paddle
import json


import yaml
from pathlib import Path
from lib.datasets import build_dataset
from supernet_engine import train_one_epoch, evaluate
from lib.samplers import RASampler
from lib import utils
from lib.config import cfg, update_config_from_file
from model.supernet_transformer import Vision_TransformerSuper
#

###
import argparse
import datetime
import json
import numpy as np
import paddle
import paddle.distributed as dist
import paddle.optimizer as optim
import paddle.nn as nn
from pathlib import Path
from lib.datasets import build_dataset
from lib.samplers import RASampler
from lib import utils
from lib.config import cfg, update_config_from_file
from model.supernet_transformer import Vision_TransformerSuper
from paddle.io import DataLoader
import paddle.nn.functional as F
from paddle.amp import GradScaler

import paddle
from paddle.amp import GradScaler
from paddle.optimizer import AdamW  # 示例优化器，根据需要替换


def tensor_to_serializable(obj):
    """递归地将 Tensor 对象转换为可以序列化的格式"""
    if isinstance(obj, paddle.Tensor):
        return obj.numpy().tolist()  # 将 Tensor 转换为列表
    elif isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_serializable(x) for x in obj]
    else:
        return obj
class NativeScaler:
    def __init__(self):
        # 在 NPU 环境下不使用 GradScaler
        self._scaler = None

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, need_update=True):
        # 在 NPU 环境下，直接进行反向传播
        loss.backward()

        if need_update:
            if clip_grad is not None :
                # 应用梯度裁剪
                 paddle.nn.ClipGradByNorm(paddle.to_tensor(clip_grad))
        else:
            # 在 NPU 环境下直接更新优化器
            optimizer.step()

        # 清除优化器的梯度
        #optimizer.clear_grad()

    def state_dict(self):
        # NPU 环境下，不保存 GradScaler 的状态
        return {}

    def load_state_dict(self, state_dict):
        # NPU 环境下，不加载 GradScaler 的状态
        pass


# class Mixup:
#     """ Mixup/Cutmix that applies different params to each element or whole batch
#
#     Args:
#         mixup_alpha (float): mixup alpha value, mixup is active if > 0.
#         cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
#         cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
#         prob (float): probability of applying mixup or cutmix per batch or element
#         switch_prob (float): probability of switching to cutmix instead of mixup when both are active
#         mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
#         correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
#         label_smoothing (float): apply label smoothing to the mixed target tensor
#         num_classes (int): number of classes for target
#     """
#     def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
#                  mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=1000):
#         self.mixup_alpha = mixup_alpha
#         self.cutmix_alpha = cutmix_alpha
#         self.cutmix_minmax = cutmix_minmax
#         if self.cutmix_minmax is not None:
#             assert len(self.cutmix_minmax) == 2
#             # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
#             self.cutmix_alpha = 1.0
#         self.mix_prob = prob
#         self.switch_prob = switch_prob
#         self.label_smoothing = label_smoothing
#         self.num_classes = num_classes
#         self.mode = mode
#         self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
#         self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)
#
#     def _params_per_elem(self, batch_size):
#         lam = np.ones(batch_size, dtype=np.float32)
#         use_cutmix = np.zeros(batch_size, dtype=bool)
#         if self.mixup_enabled:
#             if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
#                 use_cutmix = np.random.rand(batch_size) < self.switch_prob
#                 lam_mix = np.where(
#                     use_cutmix,
#                     np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size),
#                     np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size))
#             elif self.mixup_alpha > 0.:
#                 lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
#             elif self.cutmix_alpha > 0.:
#                 use_cutmix = np.ones(batch_size, dtype=bool)
#                 lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size)
#             else:
#                 assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
#             lam = np.where(np.random.rand(batch_size) < self.mix_prob, lam_mix.astype(np.float32), lam)
#         return lam, use_cutmix
#
#     def _params_per_batch(self):
#         lam = 1.
#         use_cutmix = False
#         if self.mixup_enabled and np.random.rand() < self.mix_prob:
#             if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
#                 use_cutmix = np.random.rand() < self.switch_prob
#                 lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
#                     np.random.beta(self.mixup_alpha, self.mixup_alpha)
#             elif self.mixup_alpha > 0.:
#                 lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
#             elif self.cutmix_alpha > 0.:
#                 use_cutmix = True
#                 lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
#             else:
#                 assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
#             lam = float(lam_mix)
#         return lam, use_cutmix
###


def rand_bbox_minmax(img_shape, minmax, count=None):
    """ Min-Max CutMix bounding-box
    Inspired by Darknet cutmix impl, generates a random rectangular bbox
    based on min/max percent values applied to each dimension of the input image.

    Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9 range for max.

    Args:
        img_shape (tuple): Image shape as tuple
        minmax (tuple or list): Min and max bbox ratios (as percent of image size)
        count (int): Number of bbox to generate
    """
    assert len(minmax) == 2
    img_h, img_w = img_shape[-2:]
    cut_h = np.random.randint(int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
    cut_w = np.random.randint(int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w
    return yl, yu, xl, xu


def rand_bbox(img_shape, lam, margin=0., count=None):
    """ Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def cutmix_bbox_and_lam(img_shape, lam, ratio_minmax=None, correct_lam=True, count=None):
    """ Generate bbox and apply lambda correction.
    """
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam

def one_hot(x, num_classes, on_value=1.0, off_value=0.0):
    # 将输入转换为整型并调整形状
    x = paddle.cast(x, dtype='int64').reshape([-1, 1])

    # 创建一个填充了 off_value 的张量
    output = paddle.full(shape=[x.shape[0], num_classes], fill_value=off_value, dtype=x.dtype)

    # 在第 1 维度上使用 on_value 替换指定位置的值
    output = paddle.scatter(output, index=x, updates=paddle.full(shape=x.shape, fill_value=on_value, dtype=x.dtype))

    return output

def mixup_target(target, num_classes, lam=1., smoothing=0.0):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value)
    return y1 * lam + y2 * (1. - lam)

class Mixup:
    """ Mixup/Cutmix that applies different params to each element or whole batch

    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
        prob (float): probability of applying mixup or cutmix per batch or element
        switch_prob (float): probability of switching to cutmix instead of mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
    """
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)

    def _params_per_elem(self, batch_size):
        lam = np.ones(batch_size, dtype=np.float32)
        use_cutmix = np.zeros(batch_size, dtype=bool)
        if self.mixup_enabled:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand(batch_size) < self.switch_prob
                lam_mix = np.where(
                    use_cutmix,
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size),
                    np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size))
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
            elif self.cutmix_alpha > 0.:
                use_cutmix = np.ones(batch_size, dtype=bool)
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = np.where(np.random.rand(batch_size) < self.mix_prob, lam_mix.astype(np.float32), lam)
        return lam, use_cutmix

    def _params_per_batch(self):
        lam = 1.
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_elem(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size)
        x_orig = x.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
        return paddle.to_tensor(lam_batch, place=x.place, dtype=x.dtype).unsqueeze(1)

    def _mix_pair(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        x_orig = x.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    x[j][:, yl:yh, xl:xh] = x_orig[i][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
                    x[j] = x[j] * lam + x_orig[i] * (1 - lam)
        lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
        lam_tensor = paddle.to_tensor(lam_batch, place=x.place, dtype=x.dtype)
        return lam_tensor.unsqueeze(1)

    def _mix_batch(self, x):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.:
            return 1.
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                x.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
            x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]
        else:
            x_flipped = paddle.flip(x, axis=[0]) * (1. - lam)
            #x.mul_(lam).add_(x_flipped)
            x = x * lam + x_flipped
        return lam
    # def _mix_batch(self, x):
    #     lam, use_cutmix = self._params_per_batch()
    #     if lam == 1.:
    #         return 1.
    #     if use_cutmix:
    #         # 假设cutmix_bbox_and_lam是自定义函数，需要将其修改为PaddlePaddle兼容的版本
    #         (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
    #             x.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
    #         # PaddlePaddle中没有直接的flip方法，需要使用transpose和reverse来实现
    #         x_flipped = paddle.transpose(x, [0, 2, 3, 1]).flip([1, 2]).transpose([0, 3, 1, 2])
    #         x[:, :, yl:yh, xl:xh] = x_flipped[:, :, yl:yh, xl:xh]
    #     else:
    #         # PaddlePaddle使用elementwise_add和elementwise_mul进行元素级操作
    #         x_flipped = x.flip([0]) * (1. - lam)
    #         x = lam * x + x_flipped
    #     return lam

    def __call__(self, x, target):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing)
        return x, target



###
def get_args_parser():
    parser = argparse.ArgumentParser('AutoFormer training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    # config file
    parser.add_argument('--cfg',help='experiment configure file name',required=True,type=str)

    # custom parameters
    parser.add_argument('--platform', default='pai', type=str, choices=['itp', 'pai', 'aml'],
                        help='Name of model to train')
    parser.add_argument('--teacher_model', default='', type=str,
                        help='Name of teacher model to train')
    parser.add_argument('--relative_position', action='store_true')
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--change_qkv', action='store_true')
    parser.add_argument('--max_relative_position', type=int, default=14, help='max distance in relative position embedding')

    # Model parameters
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')
    # AutoFormer config
    parser.add_argument('--mode', type=str, default='super', choices=['super', 'retrain'], help='mode of AutoFormer')
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    parser.add_argument('--rpe_type', type=str, default='bias', choices=['bias', 'direct'])
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--no_abs_pos', action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--lr-power', type=float, default=1.0,
                        help='power of the polynomial lr scheduler')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')


    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default='./data/imagenet/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='CIFAR100', choices=['CIFAR10', 'CIFAR100', 'IMNET', 'INAT', 'INAT19','GTSRB','FGSCR42','MASTER'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='./',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    parser.set_defaults(amp=True)


    return parser

#因为没sampler 自定义dataloader
class DataLoader(paddle.io.DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=1,
                 collate_fn=None,
                 pin_memory=False,
                 drop_last=False,
                 timeout=0,
                 worker_init_fn=None,
                 multiprocessing_context=None,
                 generator=None):
        if isinstance(dataset[0], (tuple, list)):
            return_list = True
        else:
            return_list = False

        super().__init__(
            dataset,
            feed_list=None,
            places=None,
            return_list=return_list,
            batch_sampler=batch_sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
            num_workers=num_workers,
            use_buffer_reader=True,
            use_shared_memory=False,
            timeout=timeout,
            worker_init_fn=worker_init_fn)
        if sampler is not None:
            self.batch_sampler.sampler = sampler


def main(args):
    #utils.init_distributed_mode(args)
    update_config_from_file(args.cfg)
    print(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    # Set device


    #device = str(args.device).replace('cuda', 'npu')
    #device='cpu'
    device='npu:0'
    paddle.set_device(device)
    #paddle.set_device('npu:1')
    # Seed
    seed = args.seed + utils.get_rank()
    paddle.seed(seed)
    np.random.seed(seed)
###
    # random.seed(seed)
    #paddle.set_flags({'FLAGS_cudnn_deterministic': False})
###
    # Build datasets
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    # Samplers and DataLoaders
    
#    if args.distributed:
#        num_tasks = utils.get_world_size()
#        global_rank = utils.get_rank()
#        if args.repeated_aug:
#            sampler_train = RASampler(dataset_train,shuffle=True)
#        else:
#            sampler_train = paddle.io.DistributedBatchSampler(dataset_train,32,
#                                                              shuffle=True)
#        if args.dist_eval:
#            if len(dataset_val) % num_tasks != 0:
#                print(
#                    'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. This will slightly alter validation results as extra duplicate entries are added to achieve equal num of samples per-process.')
#            sampler_val = paddle.io.DistributedBatchSampler(dataset_val,32,
#                                                            shuffle=False)
#        else:
#            sampler_val = paddle.io.SequenceSampler(data_source=dataset_val)
#    else:

#        sampler_val = paddle.io.SequenceSampler(dataset_val)
#        sampler_train = paddle.io.RandomSampler(dataset_train)
        
    ###
    sampler_val = paddle.io.SequenceSampler(dataset_val)
    sampler_train = paddle.io.RandomSampler(dataset_train)
###问题可能出现处
    data_loader_train = DataLoader(dataset_train,
                                   sampler=sampler_train,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   drop_last=True)

    data_loader_val = DataLoader(dataset_val,
                                 batch_size=int(2 * args.batch_size),
                                 num_workers=args.num_workers,
                                 drop_last=False,sampler=sampler_val,pin_memory=args.pin_mem)
###
    mixup_fn = None
    mixup_active = (args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None)
    # if mixup_active:
    #     mixup_fn = Mixup(
    #         mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
    #         prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
    #         label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print('Creating SuperVisionTransformer')
    print(cfg)
    model = Vision_TransformerSuper(img_size=args.input_size,
                                    patch_size=args.patch_size,
                                    embed_dim=cfg.SUPERNET.EMBED_DIM,
                                    depth=cfg.SUPERNET.DEPTH,
                                    num_heads=cfg.SUPERNET.NUM_HEADS,
                                    mlp_ratio=cfg.SUPERNET.MLP_RATIO,
                                    qkv_bias=True,
                                    drop_rate=args.drop,
                                    drop_path_rate=args.drop_path,
                                    gp=args.gp,
                                    num_classes=args.nb_classes,
                                    max_relative_position=args.max_relative_position,
                                    relative_position=args.relative_position,
                                    change_qkv=args.change_qkv,
                                    abs_pos=not args.no_abs_pos)

    choices = {'num_heads': cfg.SEARCH_SPACE.NUM_HEADS,
               'mlp_ratio': cfg.SEARCH_SPACE.MLP_RATIO,
               'embed_dim': cfg.SEARCH_SPACE.EMBED_DIM,
               'depth': cfg.SEARCH_SPACE.DEPTH}
    ###
    #dp_layer = paddle.DataParallel(model)
    ###
    model.to(device)

    if args.teacher_model:
        # Paddle does not have a direct replacement for timm's create_model method. You need to implement a function to load teacher model if necessary.
        teacher_model = None
        teacher_loss = None
    else:
        teacher_model = None
        teacher_loss = None

    model_ema = None
    model_without_ddp = model

#    if args.distributed:
#        model = paddle.DataParallel(model,find_unused_parameters=True)
#        model_without_ddp = model.module
#        # model_without_ddp = model._layers

    n_parameters = sum(p.numel() for p in model.parameters() if not p.stop_gradient)
    print('number of params:', n_parameters)

    # Learning rate scaling
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    # optimizer = optim.AdamW(learning_rate=args.lr,
    #                         parameters=dp_layer.parameters(),
    #                         weight_decay=args.weight_decay,
    #                         beta1=0.9,
    #                         beta2=0.999)
    optimizer = optim.AdamW(learning_rate=args.lr,
                            parameters=model.parameters(),
                            weight_decay=args.weight_decay,
                            beta1=0.9,
                            beta2=0.999)


    ###
    #loss_scaler = GradScaler(init_loss_scaling=1024.0)


    loss_scaler = NativeScaler()

    # Replace timm's NativeScaler with Paddle's native approach if needed
    # Replace timm's scheduler with Paddle's approach
    #lr_scheduler = optim.lr.CosineAnnealingDecay(learning_rate=args.lr, T_max=args.epochs)
    lr_scheduler = paddle.optimizer.lr.StepDecay(
        learning_rate=args.lr,
        step_size=1000,  # 每1000步调整一次学习率
        gamma=0.1  # 每次调整学习率乘以0.1
    )



    # if args.mixup > 0.0:
    #     #criterion = nn.SoftmaxCrossEntropyWithLogits()
    #     criterion = nn.CrossEntropyLoss(reduction='mean', soft_label=True)
    # elif args.smoothing:
    #     criterion = nn.CrossEntropyLoss()
    # else:
    #     criterion = nn.CrossEntropyLoss()
    if args.mixup > 0.0:
        #criterion = nn.SoftmaxCrossEntropyWithLogits()
        criterion = nn.CrossEntropyLoss(reduction='mean', soft_label=False)
    elif args.smoothing:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()


    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    with open(output_dir / 'config.yaml', 'w') as f:
        f.write(args_text)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = paddle.load(args.resume)
        else:
            checkpoint = paddle.load(path=args.resume)

        model_without_ddp.set_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            optimizer.set_state_dict(checkpoint['optimizer'])
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.set_state_dict(checkpoint['lr_scheduler'])
        if 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch'] + 1
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        if args.model_ema:
            utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])

    retrain_config = None
    if args.mode == 'retrain' and 'RETRAIN' in cfg:
        retrain_config = {'layer_num': cfg.RETRAIN.DEPTH,
                          'embed_dim': [cfg.RETRAIN.EMBED_DIM] * cfg.RETRAIN.DEPTH,
                          'num_heads': cfg.RETRAIN.NUM_HEADS,
                          'mlp_ratio': cfg.RETRAIN.MLP_RATIO}

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, mode=args.mode, retrain_config=retrain_config)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print('Start training')
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
#        if args.distributed:
#            data_loader_train.batch_sampler.set_epoch(epoch)


        # 调用train_one_epoch函数
        # train_stats = train_one_epoch(
        #     model=model,
        #     criterion=criterion,
        #     data_loader=data_loader_train,
        #     optimizer=optimizer,
        #     device=device,
        #     epoch=epoch,
        #     loss_scaler=None,  # 如果有需要loss_scaler, 请替换为实际对象
        #     max_norm=0,  # 如果需要梯度裁剪，可以调整此参数
        #     model_ema=model_ema,
        #     mixup_fn=mixup_fn,
        #     amp=True,  # 如果需要自动混合精度，可以设置为True，否则设置为False
        #     teacher_model=teacher_model,
        #     teach_loss=teacher_loss,
        #     choices=choices,
        #     mode=args.mode,
        #     retrain_config=retrain_config
        # )
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            amp=args.amp, teacher_model=teacher_model,
            teach_loss=teacher_loss,
            choices=choices, mode = args.mode, retrain_config=retrain_config,
        )
        lr_scheduler.step(epoch)

        # if args.output_dir:
        #     checkpoint_paths = [output_dir / 'checkpoint.pdparams']
        #     for checkpoint_path in checkpoint_paths:
        #         utils.save_on_master({'model': model_without_ddp.state_dict(),
        #                               'optimizer': optimizer.state_dict(),
        #                               'lr_scheduler': lr_scheduler.state_dict(),
        #                               'epoch': epoch,
        #                               'args': args}, checkpoint_path)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    # 'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
        test_stats = evaluate(data_loader_val, model, device, amp=args.amp, choices=choices, mode=args.mode,
                              retrain_config=retrain_config)
        # test_stats = evaluate(data_loader_val, model, device, choices=choices, mode=args.mode,
        #                       retrain_config=retrain_config)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats['acc1'])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        # if args.output_dir and utils.is_main_process():
        #     with (output_dir / 'log.txt').open('a') as f:
        #         f.write(json.dumps(log_stats) + '\n')

        if args.output_dir and utils.is_main_process():
            # 将 log_stats 转换为可以序列化的格式
            serializable_log_stats = tensor_to_serializable(log_stats)

            # 将转换后的字典写入 JSON 文件
            with (output_dir / 'log.txt').open('a') as f:
                f.write(json.dumps(serializable_log_stats) + '\n')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'AutoFormer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)