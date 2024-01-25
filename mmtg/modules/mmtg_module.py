import os
import torch
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from .input2emb import MMTGInput2Emb

class MMTGTransformer(nn.Module):
    def __init__(self, _config):
        super().__init__()

        self.test_only = _config['test_only']
        self.nc = _config['nc']
        self.nn = _config['nn']
        self.ntable = self.nc + self.nn
        self.ntext = _config['ntext']
        self.use_table = 'table' in _config['modality'] 
        self.temperature = _config['temperature']
        self.topk_filter_thres = _config['topk_filter_thres']
        self.top_k = _config['topk']
        self.top_p = _config['topp']

        bert_config = {
            "tiny": (4, 128, 4),
            "mini": (4, 256, 4),
            "small": (6, 256, 4),
            "medium": (8, 512, 8),
            "base": (12, 768, 12),
        }

        num_layers, self.embed_dim, n_head = bert_config[_config['bert_type']]
        encoder_layer = TransformerEncoderLayer(
            dropout=_config["dropout"],
            d_model=self.embed_dim,
            nhead=n_head,
            dim_feedforward=4*self.embed_dim,
            activation='gelu',
            layer_norm_eps=1e-12,
            batch_first=True
        )
        
        self.bert = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.input2emb = MMTGInput2Emb(_config)
        self.emb2out = nn.Linear(self.embed_dim, _config['text_vocab_size'])

        self.soe_token = _config["soe_token"]
        self.eos_token = _config["eos_token"]
        
        self.prevent_repeat_ngram = _config["prevent_repeat_ngram"]
        self.prevent_too_short = _config["prevent_too_short"]

    def forward(self, **kwargs):
        x = self.input2emb(**kwargs)
        device = x.device
        text_mask = self._generate_square_subsequent_mask(self.ntext).to(device)
        if self.use_table:
            table_mask = torch.zeros(self.ntable, self.ntable) # Table tokens can reference each other
            table_to_text_mask = torch.ones(self.ntable, self.ntext) * float('-inf') # Table tokens cannot reference text tokens
            text_to_table_mask = torch.zeros(self.ntext, self.ntable) # Text tokens can reference table tokens

            top = torch.cat([table_mask.to(device), table_to_text_mask.to(device)], dim=1)
            bottom = torch.cat([text_to_table_mask.to(device), text_mask], dim=1)
            enc_attention_mask = torch.cat([top, bottom], dim=0)
        else:
            enc_attention_mask = text_mask

        x = self.bert(x, enc_attention_mask)
        x = self.emb2out(x)

        if self.use_table:
            x = x[:, self.ntable:, :]
            
        return x[:, :-1, :]

    def generate(self, **kwargs):
        text_input_ids = torch.ones(len(kwargs["text_input_ids"]), 1).fill_(self.soe_token).long().cuda()
        
        for i in tqdm(range(self.ntext - 1)):
            kwargs["text_input_ids"] = text_input_ids
            kwargs["text_type_ids"] = torch.ones_like(text_input_ids).to(text_input_ids.device)
            
            x = self.input2emb(**kwargs)
            logits = self.emb2out(self.bert(x))[:, -1, :]
            logits = logits / self.temperature
            
            if i < self.prevent_too_short: # 10 percentile
                logits = self._mask_eos_token_logits(logits)
            if self.prevent_repeat_ngram != 0:
                logits = self._prevent_repeat_bigram(logits, text_input_ids, n=self.prevent_repeat_ngram)
            if self.top_p is not None:
                logits = self._top_p_logits(logits, self.top_p)

            probs = F.softmax(logits, dim=-1)
            
            next_word = self._get_next_word(probs, sample=True)
            text_input_ids = torch.cat([text_input_ids, next_word], dim=1)

        end_positions = (text_input_ids == self.eos_token).nonzero(as_tuple=True)
        for batch_index, seq_index in zip(*end_positions):
            text_input_ids[batch_index, seq_index+1:] = 0

        return text_input_ids
    
    def _prevent_repeat_bigram(self, logits, text_input_ids, n=3):
        """
        Set the probability of the last generated token to negative infinity to prevent its repetition.
        This is only applied before the end token is encountered.
        """
        # Check if end token is already in the text_input_ids
        if self.eos_token not in text_input_ids:
            if text_input_ids.shape[1] > 1:
                logits = logits.clone()

                for i in range(logits.size(0)):
                    last_tokens = text_input_ids[i, -n:]
                    logits[i, last_tokens] = logits[i, last_tokens] - 1e5 
        
        return logits

    def _mask_eos_token_logits(self, logits):
        logits[:, self.eos_token] = -float('Inf')
        return logits
            
    def _top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    def _top_p_logits(self, logits, p):
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Sort the probabilities and compute cumulative probabilities
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find the point where cumulative probability exceeds p
        # Set all values from that point to the end to 0 (masking)
        sorted_indices_to_remove = cumulative_probs > p
        # Ensure all values after the first point where cumulative probability exceeds p are set to True
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Rearrange the indices to restore the original order
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
        return logits

    def _generate_square_subsequent_mask(self, sz: int):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    
    def _get_next_word(self, probs, sample):
        """Sample or get the top probability word."""
        if sample:
            return torch.multinomial(probs.squeeze(0), num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
            return ix.squeeze(0)
    
    def get_targets(self, sample):
        targets = {}
        targets["text_input_ids"] = sample['net_input']['text_input_ids'][:, 1:]
        return targets
    