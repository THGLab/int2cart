"""
This script defines a wrapper that converts a standard data loader into an infinite one
"""
import random

class InfiniteDataLoader():
    def __init__(self, data_loader) -> None:
        self.contents = []
        for item in data_loader:
            self.contents.append(item)
        self.indices = list(range(len(self.contents)))
        self.iterator = iter(self)

    def __iter__(self):
        while 1:
            random.shuffle(self.indices)
            for i in self.indices:
                yield self.contents[i]

    def __next__(self):
        return next(self.iterator)

if __name__ == '__main__':
    import sidechainnet as scn
    container = scn.load("debug",
            with_pytorch='dataloaders',
            scn_dir="/home/jerry/data2/protein_building/sidechainnet_data",
            batch_size=4,)
    loader = InfiniteDataLoader(container["test"])  
    i=0
    while 1:
        a=next(loader)   
        print(a.pids)       
        i+=1
        print(i)