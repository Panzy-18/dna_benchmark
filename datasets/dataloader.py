import torch.utils.data as data
import bmtrain as bmt

class DistributedDataLoader:
    def __init__(self, dataset, shuffle=False, **kwargs):
        self.dataset = dataset
        self.shuffle = shuffle
        self.kwargs = kwargs
        self.epoch = 0
        self.__post_init__()
    
    def __post_init__(self):
        self.sampler = data.DistributedSampler(self.dataset, shuffle=self.shuffle, rank=bmt.rank(), num_replicas=bmt.world_size())
        self.loader = data.DataLoader(self.dataset, shuffle=False, sampler=self.sampler, **self.kwargs)

    def reload(self):
        if hasattr(self.dataset, 'reload'):
            self.dataset.reload()
            self.__post_init__()

    def __iter__(self):
        if self.shuffle:
            self.epoch += 1
        self.sampler.set_epoch(self.epoch)
        return self.loader.__iter__()

    def __len__(self):
        return len(self.loader)
