import torch
import torch.nn as nn
import math


class MMTGInput2Emb(nn.Module):
    def __init__(self, _config):
        super().__init__()
        
        self.use_table = 'table' in _config['modality'] 
        self.ntext = _config['ntext']

        bert_config = {
            "tiny": (4, 128, 4),
            "mini": (4, 256, 4),
            "small": (6, 256, 4),
            "medium": (8, 512, 8),
            "base": (12, 768, 12),
        }

        _, self.emb_dim, _ = bert_config[_config['bert_type']]

        self.sep_numemb = _config['sep_numemb'] 
        if self.use_table:
            self.cat_emb = nn.Embedding(_config['cat_vocab_size'], self.emb_dim)
            self.num_emb = nn.Parameter(torch.randn(_config['nn'], self.emb_dim)) if self.sep_numemb else nn.Linear(1, self.emb_dim)
        else:
            self.cat_emb = None
            self.num_emb = None
        self.text_emb = nn.Embedding(_config['text_vocab_size'], self.emb_dim, padding_idx=0)

        """
        Positional Encoding
        - Table: Column-wise Learnable Embedding
        - Text: Sinusoidal Positional Encoding
        """
        self.cat_pos_emb = nn.Parameter(torch.randn(_config['nc'], self.emb_dim)) if self.use_table else None
        self.num_pos_emb = nn.Parameter(torch.randn(_config['nn'], self.emb_dim)) if self.use_table else None
        self.text_pos_emb = PositionalEncoding(self.emb_dim, _config['dropout'], _config['ntext']) 

        """
        Token type Embedding: Table vs Text
        """
        self.token_type_emb = nn.Embedding(2, self.emb_dim)

        """
        Null type Embedding
        - 0: not null
        - 1: null
        """
        self.null_emb = nn.Embedding(2, self.emb_dim)


    def forward(
        self,
        text_input_ids,
        text_type_ids,
        cat_input_ids=None,
        cat_null_ids=None,
        cat_type_ids=None,
        num_input_ids=None,
        num_null_ids=None,
        num_type_ids=None,
        **kwargs
    ): 
        #Embeddings
        device = text_input_ids.device
        text_emb = self.text_emb(text_input_ids)
        ntext= text_emb.shape[1]

        pe_index = torch.arange(0, ntext).expand(text_emb.shape[0], ntext).to(text_emb.device, dtype=torch.long)
        text_emb = self.text_pos_emb(text_emb, pe_index) 
        text_emb += self.token_type_emb(text_type_ids)
        
        if self.use_table:

            cat_input_ids = cat_input_ids.to(device)
            cat_null_ids = cat_null_ids.to(device)
            cat_type_ids = cat_type_ids.to(device)
            num_input_ids = num_input_ids.to(device)
            num_null_ids = num_null_ids.to(device)
            num_type_ids = num_type_ids.to(device)

            cat_emb = self.cat_emb(cat_input_ids)
            cat_emb += self.cat_pos_emb
            cat_emb += self.token_type_emb(cat_type_ids)
            cat_emb += self.null_emb(cat_null_ids)
            
            if self.sep_numemb:
                num_input_ids = num_input_ids.unsqueeze(-1)
                B, S, _ = num_input_ids.shape
                num_emb = torch.mul(num_input_ids.expand(B, S, self.emb_dim), self.num_emb)
            else:
                num_emb = self.num_emb(num_input_ids.unsqueeze(-1))
            num_emb += self.num_pos_emb
            num_emb += self.token_type_emb(num_type_ids)
            num_emb += self.null_emb(num_null_ids)
            return torch.cat([cat_emb, num_emb, text_emb], axis=1)
        
        else:
            return text_emb


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dimension, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        max_len = int(max_len)

        pe = torch.zeros(max_len, embedding_dimension)   
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dimension, 2).float() * (-math.log(10000.0) / embedding_dimension))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.positional_embed = nn.Embedding(int(max_len), embedding_dimension, _weight=pe)
        
    def forward(self, x, pe_index):

        with torch.no_grad():
            positional_embed = self.positional_embed(pe_index) #(B, S, E)
        x = x + positional_embed 

        output = self.dropout(x)
        return output
