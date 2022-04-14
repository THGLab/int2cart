'''
This script defines a wrapper class that combines multiple data loaders into one sequential data loader
'''


class MultipleDataLoader():
    def __init__(self, data_loaders) -> None:
        self.data_loaders = data_loaders
        self.iterator = iter(self)

    def __iter__(self):
        for loader in self.data_loaders:
            for content in loader:
                yield content

    def __next__(self):
        return next(self.iterator)

    def __len__(self):
        return sum([len(item) for item in self.data_loaders])