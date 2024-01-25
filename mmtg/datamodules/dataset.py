import os
import random
import torch
import math
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
# from transformers import BertTokenizer

class MMTGDataset(torch.utils.data.Dataset):
    def __init__(self, _config, split):
        super().__init__()

        self.split = split 
        self.seed = _config['seed']
        self.nc, self.nn, self.ntext = _config['nc'], _config['nn'], _config['ntext']
        self.ntable = self.nc + self.nn
        self.use_table = 'table' in _config['modality'] 

        # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        # tokenizer.add_special_tokens({'eos_token': '[EOS]'}) # 119547
        self.sep_token = _config["sep_token"]
        self.eos_token = _config["eos_token"]

        data_root = _config['input_path']
        self.fold = os.path.join(data_root, f'fold/snuh_{self.seed}_fold_split.csv')  
        hit_idcs = self._get_fold_indices()
        
        self.input_data = np.load(os.path.join(data_root, 'input_ids.npy'))[hit_idcs]
        self.null_type_data = np.load(os.path.join(data_root, 'null_type_ids.npy'))[hit_idcs]
        self.data_dict = self._get_data()

        self.test_only = _config['test_only']
        self.modeling = _config['modeling']


    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        text_data = self._prepare_text(index)
        # Include textual data if required
        if self.use_table:
            cat_data = self._prepare_data('cat', index)
            num_data = self._prepare_data('num', index)
            return {'cat': cat_data, 'num': num_data, 'text': text_data}
        else:
            return {'text': text_data}

    def collator(self, samples):
        input, out = {}, {}

        input_types = ['cat', 'num', 'text'] if self.use_table else ['text']

        # Collate samples
        for input_type in input_types:
            input = self._collate_samples(input, samples, input_type)

        out["net_input"] = input
        return out

  

    def _prepare_data(self, data_type, index):
        if data_type == 'cat':
            input_ids = torch.LongTensor(self.data_dict[data_type]['input'][index])
        elif data_type == 'num':
            input_ids = torch.FloatTensor(self.data_dict[data_type]['input'][index])

        null_type = torch.LongTensor(self.data_dict[data_type]['null'][index])
        token_type = torch.LongTensor(self.data_dict[data_type]['type'][index])
                
        return {
            'input_ids':input_ids,
            'null_ids':null_type,
            'type_ids':token_type,
        }
    
    def _prepare_text(self, index):
        input_ids = torch.LongTensor(self.data_dict['text']['input'][index])
        token_type = torch.LongTensor(self.data_dict['text']['type'][index])
        
        return {
            'input_ids':input_ids,
            'type_ids':token_type,
        }


    def _collate_samples(self, input, samples, input_type):
        input[f"{input_type}_input_ids"] = torch.stack([s[input_type]["input_ids"] for s in samples])
        input[f"{input_type}_type_ids"] = torch.stack([s[input_type]["type_ids"] for s in samples])
        if input_type != 'text':
            for key in ["null_ids"]:
                input[f"{input_type}_{key}"] = torch.stack([s[input_type][key] for s in samples])
        return input
    
    def _get_data(self): 
        # Define data types and their indices
        data_types = {
            'cat': (0, self.nc),
            'num': (self.nc, self.ntable)
        }

        # Include 'text' data type if required
        data_types['text'] = (self.ntable, None)
    
        data = {}
        for data_type, (start_idx, end_idx) in data_types.items():
            # Slice input data according to start and end indices
            input_data = self.input_data[:, start_idx:end_idx]
            
            # Prepare null_type data. Only for 'cat' and 'num' data types.
            null_type = self.null_type_data[:, start_idx:end_idx] if data_type != 'text' else None
            
            # If data type is 'text', token type is all ones. Otherwise, it's all zeros.
            token_type = np.ones_like(input_data) if data_type == 'text' else np.zeros_like(input_data)

            data[data_type] = {
                'input': input_data,
                'null': null_type,
                'type': token_type
            }
            
        # Refine text data
        for row in  data["text"]["input"]:
            last_sep_indices = np.where(row == self.sep_token)[0]
            if last_sep_indices.size > 0:
                row[last_sep_indices[-1]] = self.eos_token
    
        return data

    def _get_fold_indices(self):
        split_idx = {'train':1, 'valid':2, 'test':0}
        hit = split_idx[self.split]

        splits = pd.read_csv(self.fold)['fold'].values
        idcs = np.where(splits == hit)[0]
        return idcs


def mmtg_data_loader(_config):
    data_loaders = dict()
    for split in ['train', 'valid', 'test']:
        dataset = MMTGDataset(_config, split)
        shuffle = True if split == 'train' else False
        data_loaders[split] = DataLoader(
            dataset, collate_fn=dataset.collator, batch_size=_config['per_gpu_batchsize'], num_workers=8, shuffle=shuffle
            )
    
    return data_loaders