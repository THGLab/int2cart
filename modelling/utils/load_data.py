"""
Data loaders for training
"""

from sidechainnet import load
from modelling.utils.infinite_data_loader import InfiniteDataLoader
from modelling.utils.multiple_data_loader import MultipleDataLoader

def load_data(settings):
    data = load(settings['data']['casp_version'],
                thinning=settings['data']['thinning'],
                with_pytorch='dataloaders',
                scn_dir=settings['data']['scn_data_dir'],
                batch_size=settings['training']['batch_size'],
                filter_by_resolution=settings['data'].get("filter_resolution", False))

    train = data['train']
    validation_similarities = settings["data"]["validation_similarity_level"]
    if type(validation_similarities) is list:
        val = MultipleDataLoader([data[f'valid-{val_level}'] for val_level in validation_similarities])
    else:
        val = data[f'valid-{validation_similarities}']
    infinite_val = InfiniteDataLoader(val)
    test = data['test']
    return train, val, test, infinite_val