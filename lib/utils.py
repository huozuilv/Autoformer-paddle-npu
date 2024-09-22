import os
import paddle
import io
import time
from collections import defaultdict, deque
import datetime
import paddle.distributed as dist
from pathlib import Path
import numpy as np


###初版可运行SmoothedValue
# class SmoothedValue:
#     def __init__(self, window_size: int = 1, fmt: str = '{value:.6f}'):
#         self.window_size = window_size
#         self.fmt = fmt
#         self.reset()
#
#     def reset(self):
#         self.values = []
#         self.total = 0
#         self.count = 0
#
#     def update(self, value, n=1):
#         self.values.append(value)
#         self.total += value * n
#         self.count += n
#         if len(self.values) > self.window_size:
#             self.total -= self.values.pop(0) * n
#
#     @property
#     def avg(self):
#         return self.total / self.count if self.count > 0 else 0
#
#     @property
#     def global_avg(self):
#         return self.total / self.count
#
#
#     def __str__(self):
#         return self.fmt.format(value=self.avg)
class SmoothedValue:
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt


    def reset(self):
        self.values = []
        self.total = 0
        self.count = 0

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        警告：不会同步 deque！
        """
        if not dist.is_initialized():
            return
        t = paddle.to_tensor([self.count, self.total], dtype='float64', place='npu:0')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = paddle.to_tensor(list(self.deque))
        return paddle.median(d).item()

    @property
    def avg(self):
        d = paddle.to_tensor(list(self.deque), dtype='float32')
        return paddle.mean(d).item()

    @property
    def global_avg(self):
        return self.total / self.count
    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)



###初版可运行MetricLogger
# class MetricLogger:
#     def __init__(self, delimiter='  '):
#         self.meters = {}
#         self.delimiter = delimiter
#
#     def add_meter(self, name, meter):
#         self.meters[name] = meter
#
#     def update(self, **kwargs):
#         for name, value in kwargs.items():
#             if name not in self.meters:
#                 self.add_meter(name, SmoothedValue())
#             self.meters[name].update(value)
#
#     def log_every(self, iterable, print_freq, header=None):
#         for i, batch in enumerate(iterable):
#             yield batch
#             if i % print_freq == 0:
#                 print(f'{header} [{i}] {self.delimiter.join(f"{name} {meter}" for name, meter in self.meters.items())}')
#
#     def synchronize_between_processes(self):
#         for meter in self.meters.values():
#             meter.synchronize_between_processes()
#
#     def __str__(self):
#         return self.delimiter.join(f"{name} {meter}" for name, meter in self.meters.items())

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    # def update(self, **kwargs):
    #     for k, v in kwargs.items():
    #         if isinstance(v, paddle.Tensor):
    #             v = v.item()
    #         assert isinstance(v, (float, int))
    #         self.meters[k].update(v)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, paddle.Tensor):
                v = v.item()  # 将 Tensor 转换为标量值
            elif isinstance(v, np.ndarray):
                if v.size == 1:  # 如果 ndarray 中只有一个元素
                    v = v.item()  # 转换为标量
                else:
                    raise ValueError(f"Expected a scalar value, but got an array for {k}.")
            assert isinstance(v, (float, int)), f"Expected float or int, but got {type(v)}"
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
#        if paddle.device:
#            log_msg.append('max mem: {memory:.0f}')
#        log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
#                if paddle.device:
#                    print(log_msg.format(
#                        i, len(iterable), eta=eta_string,
#                        meters=str(self),
#                        time=str(iter_time), data=str(data_time),memory=0
#                        ))
#                else:
#                    print(log_msg.format(
#                        i, len(iterable), eta=eta_string,
#                        meters=str(self),
#                        time=str(iter_time), data=str(data_time)))
                print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))





def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    paddle.save(obj=checkpoint, path=mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not paddle.distributed.is_available():
        return False
    if not paddle.distributed.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return paddle.distributed.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return paddle.distributed.get_rank()


def is_main_process():
    return get_rank() == 0


# def save_on_master(*args, **kwargs):
#     if is_main_process():
#        paddle.save(*args, **kwargs)

def save_on_master(*args, **kwargs):
    if is_main_process():
        # 检查路径参数并将其转换为字符串
        if 'path' in kwargs:
            if isinstance(kwargs['path'], Path):
                kwargs['path'] = str(kwargs['path'])
        elif len(args) > 1 and isinstance(args[1], Path):
            args = list(args)
            args[1] = str(args[1])
            args = tuple(args)
        paddle.save(*args, **kwargs)

def init_distributed_mode(args):
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        args.rank = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
        args.world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE'))
        args.npu = args.rank % paddle.device.get_device().count()  # 获取 NPU 数量
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.npu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])  # 使用 SLURM 环境中的任务数作为 world_size
        args.npu = args.rank % paddle.device.get_device().count()  # 获取 NPU 数量
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    # 设置 NPU 设备
    paddle.device.set_device(device='npu:' + str(args.npu))
    
    # 确认使用 HCCL 后端进行分布式通信
    args.dist_backend = 'hccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)

    # 初始化分布式环境
    paddle.distributed.init_parallel_env()
    paddle.distributed.barrier()

    # 仅在 rank 为 0 的进程中设置打印行为
    setup_for_distributed(args.rank == 0)

