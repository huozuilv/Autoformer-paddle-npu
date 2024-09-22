import paddle
import math


class RASampler(paddle.io.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not paddle.distributed.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            num_replicas = paddle.distributed.get_world_size()
        if rank is None:
            if not paddle.distributed.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            rank = paddle.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 3.0 / self.
            num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 *
            256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        #原本g = torch.Generator()
        g = paddle.framework.core.default_cpu_generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = paddle.randperm(n=len(self.dataset)).tolist()
        else:
            indices = list(range(len(self.dataset)))
        indices = [ele for ele in indices for i in range(3)]
        indices += indices[:self.total_size - len(indices)]
        assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
