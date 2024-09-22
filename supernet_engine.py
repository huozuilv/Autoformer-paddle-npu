import datetime
import time

import paddle
import math
import sys
import random
from typing import Iterable, Optional
from paddle.vision.transforms import Compose, Normalize, Resize, ToTensor
from collections import defaultdict, deque
import paddle.distributed as dist
from lib import utils
import paddle
import numpy as np

###

import paddle.nn as nn
import paddle.optimizer as opt
import paddle.distributed as dist

###

# def unwrap_model(model):
#     return model
def unwrap_model(model):
    if isinstance(model, ModelEma):
        return unwrap_model(model.ema)
    else:
        return model.module if hasattr(model, 'module') else model

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]

    # 获取前 maxk 个预测结果
    _, pred = paddle.topk(output, k=maxk, axis=1)
    pred = pred.astype('int64')

    # 计算预测与真实标签的匹配情况
    correct = pred == target.unsqueeze(1)

    res = []
    for k in topk:
        if correct.ndim == 1:
            correct = correct.unsqueeze(1)
        correct_k = paddle.sum(correct[:, :k], axis=1)
        # res.append(correct_k.astype('float32').mean().numpy())
        ###
        accuracy_k = correct_k.astype('float32').mean().numpy() * 100
        res.append(accuracy_k)
        ###
    return res

# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     maxk = min(max(topk), output.size()[1])
#     batch_size = target.size(0)
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.reshape(1, -1).expand_as(pred))
#     return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def sample_configs(choices):
    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['depth'])
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]
    config['embed_dim'] = [random.choice(choices['embed_dim'])] * depth
    config['layer_num'] = depth
    # 添加 sample_in_dim 和 sample_out_dim
    config['sample_in_dim'] = config['embed_dim'][0]  # 使用 embed_dim 作为 sample_in_dim
    config['sample_out_dim'] = config['embed_dim'][0]  # 确保 sample_out_dim 和 sample_in_dim 一致
    return config

###




class Mixup:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.shape[0]
        index = paddle.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index]

        return mixed_x, mixed_y



class ModelEma:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow_params = {name: param.clone() for name, param in model.named_parameters()}

    def update(self):
        for name, param in self.model.named_parameters():
            self.shadow_params[name] = self.decay * self.shadow_params[name] + (1.0 - self.decay) * param

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            param.set_value(self.shadow_params[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            self.shadow_params[name], param = param, self.shadow_params[name]
###

def train_one_epoch(model: paddle.nn.Layer, criterion: paddle.nn.Layer,
                    data_loader: Iterable, optimizer: paddle.optimizer.Optimizer,
                    device: (paddle.CPUPlace, paddle.CUDAPlace, str), epoch: int, loss_scaler,
                    max_norm: float = 0, model_ema: Optional[ModelEma] = None,
                    mixup_fn: Optional[Mixup] = None, amp: bool = True,
                    teacher_model: paddle.nn.Layer = None, teach_loss: paddle.nn.Layer = None,
                    choices=None, mode='super', retrain_config=None):
    model.train()
    criterion.train()
    random.seed(epoch)
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        print(config)
        model_module.set_sample_config(config=config)
        print(model_module.get_sampled_params_numel(config))
    #####
    #device = 'npu:1'  # 设置为第一个 NPU 设备
    #paddle.set_device(device)
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = paddle.to_tensor(samples, place=device)
        targets = paddle.to_tensor(targets, place=device)
        if mode == 'super':
            config = sample_configs(choices=choices)
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        elif mode == 'retrain':
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)

        # 调试信息
        # print(f"samples shape: {samples.shape}")
        # print(f"Expected sample_in_dim: {model_module.sample_in_dim}, sample_out_dim: {model_module.sample_out_dim}")

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            ###
            #targets = paddle.argmax(targets, axis=1)
            ###
        if amp and device=='gpu':
            with paddle.amp.auto_cast():
                if teacher_model:
                    with paddle.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = paddle.topk(teach_output, k=1, axis=1)
                    outputs = model(samples)
                    loss = 0.5 * criterion(outputs, targets) + 0.5 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    outputs = model(samples)
                    loss = criterion(outputs, targets)
                    #loss = paddle.nn.functional.cross_entropy(outputs, targets, soft_label=True)

        else:
            outputs = model(samples)
            if teacher_model:
                with paddle.no_grad():
                    teach_output = teacher_model(samples)
                _, teacher_label = paddle.topk(teach_output, k=1, axis=1)
                loss = 0.5 * criterion(outputs, targets) + 0.5 * teach_loss(outputs, teacher_label.squeeze())
            else:
                loss = criterion(outputs, targets)
        loss_value = loss.numpy()
        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            sys.exit(1)
        optimizer.clear_gradients()
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

            loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(),
                        create_graph=is_second_order)
        else:
            loss.backward()
            optimizer.step()
        #paddle.device.synchronize()
        if model_ema is not None:
            model_ema.update()
            #model_ema.update(model)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.get_lr())
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@paddle.no_grad()
def evaluate(data_loader, model, device, amp=True, choices=None, mode='super', retrain_config=None):
    criterion = paddle.nn.CrossEntropyLoss()
    metric_logger =utils.MetricLogger(delimiter='  ')
    ###
    metric_logger.add_meter('acc1', utils.SmoothedValue(window_size=1, fmt='{value:.3f}'))
    metric_logger.add_meter('acc5', utils.SmoothedValue(window_size=1, fmt='{value:.3f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.3f}'))
    ###
    header = 'Test:'
    model.eval()
    if mode == 'super':
        config = sample_configs(choices=choices)
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    print('sampled model config: {}'.format(config))
    parameters = model_module.get_sampled_params_numel(config)
    print('sampled model parameters: {}'.format(parameters))
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = paddle.to_tensor(images, place=device)
        target = paddle.to_tensor(target, place=device)
        if amp and device=='gpu':
            with paddle.amp.auto_cast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        #
        #
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.numpy())
        metric_logger.meters['acc1'].update(acc1, n=batch_size)
        metric_logger.meters['acc5'].update(acc5, n=batch_size)
    metric_logger.synchronize_between_processes()
    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'.format(
    #     top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    ###
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'.format(
        top1=metric_logger.meters['acc1'], top5=metric_logger.meters['acc5'], losses=metric_logger.meters['loss']))
    ###
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
