'''
This file defines a wrapper class for the SidechainNet data loader to generate data on a single residue/tripeptide level
'''
import random
from math import ceil
import torch
from sidechainnet.dataloaders.collate import Batch

class PeptideFragmentLoader():
    def __init__(self, data_loader, fragment_len, batch_size) -> None:
        self.data_loader = data_loader
        self.fragment_len = fragment_len
        self.batch_size = batch_size

    def __iter__(self):
        for item in self.data_loader:
            # prepare all indices for this batch of data
            needed_item_indices = []
            for idx, length in enumerate(item.lengths):
                for pos in range(length - self.fragment_len + 1):
                    if item.msks[idx, pos:pos + self.fragment_len].all():
                        needed_item_indices.append((idx, pos))

            # shuffle indices and start generation
            random.shuffle(needed_item_indices)
            for i in range(ceil(len(needed_item_indices) / self.batch_size)):
                indices_this_batch = needed_item_indices[i * self.batch_size: (i + 1) * self.batch_size]
                batch_dict = {}
                for cat in ['pids', 'resolutions']:
                    batch_dict[cat] = [getattr(item, cat)[idx[0]] for idx in indices_this_batch]
                for cat in ['angs', 'blens', 'evos', 'secs', 'seq_evo_sec', 'seqs']:
                    batch_dict[cat] = torch.zeros((len(indices_this_batch), self.fragment_len, getattr(item, cat).shape[-1]))
                    for batch_i in range(len(indices_this_batch)):
                        bi, pos = indices_this_batch[batch_i]
                        batch_dict[cat][batch_i, :] = getattr(item, cat)[bi, pos: pos + self.fragment_len]
                batch_dict['crds'] = torch.zeros((len(indices_this_batch), self.fragment_len * 14, 3))
                for batch_i in range(len(indices_this_batch)):
                    bi, pos = indices_this_batch[batch_i]
                    batch_dict['crds'][batch_i, :] = item.crds[bi, pos * 14: (pos + self.fragment_len) * 14]
                for cat in ['int_seqs', 'is_modified']:
                    batch_dict[cat] = torch.zeros((len(indices_this_batch), self.fragment_len))
                    for batch_i in range(len(indices_this_batch)):
                        bi, pos = indices_this_batch[batch_i]
                        batch_dict[cat][batch_i, :] = getattr(item, cat)[bi, pos: pos + self.fragment_len]


                batch = Batch(pids=batch_dict['pids'],
                              seqs=batch_dict['seqs'],
                              msks=torch.ones((len(indices_this_batch), self.fragment_len)),
                              evos=batch_dict['evos'],
                              secs=batch_dict['secs'],
                              angs=batch_dict['angs'],
                              crds=batch_dict['crds'],
                              int_seqs=batch_dict['int_seqs'],
                              seq_evo_sec=batch_dict['seq_evo_sec'],
                              resolutions=batch_dict['resolutions'],
                              is_modified=batch_dict['is_modified'],
                              lengths=[self.fragment_len] * len(indices_this_batch),
                              str_seqs=[item.str_seqs[bi][pos: pos + self.fragment_len] for bi, pos in indices_this_batch],
                              blens=batch_dict['blens'])
                yield batch


if __name__ == '__main__':
    import sidechainnet as scn
    container = scn.load("debug",
            with_pytorch='dataloaders',
            scn_dir="/home/jerry/data2/protein_building/sidechainnet_data",
            batch_size=1,)
    loader = PeptideFragmentLoader(container["train"],1,256)
    counter=0
    for item in loader:
        counter+=1
    print(counter)