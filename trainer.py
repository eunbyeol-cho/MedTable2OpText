import os
import tqdm
import logging
import wandb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel.data_parallel import DataParallel
from mmtg.criterions import ARCriterion
import mmtg.utils.trainer_utils as utils

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, model, data_loaders, _config):

        self.config = _config
        
        self.model = nn.DataParallel(model).to('cuda')
        logger.info(self.model)
        self.data_loaders = data_loaders
        utils.count_parameters(self.model)

        #Training settings
        self.criterion = ARCriterion(_config)
        self.n_epochs = self.config['n_epochs']
        self.lr =  self.config['lr']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.patience = self.config['patience']

        exp_component = [
            ''.join(self.config['modality']),
            self.config['modeling'],
            self.config['bert_type'],
            self.config['var_head_type'],
            self.config['sep_regressor'],
            self.config['sep_numemb'],
            self.config['lr'],
            self.config['dropout'],
            self.config['seed'],
            ]

        ckpt_name = '_'.join(list(map(str, exp_component)))
        self.path = os.path.join(self.config['output_path'], ckpt_name)

        #Wandb
        if not self.config['debug']:
            wandb.init(
                project=self.config['wandb_project_name'],
                entity="emrsyn",
                config=self.config,
                reinit=True
            )
            wandb.run.name = ckpt_name

    def train(self):
        self.early_stopping = utils.EarlyStopping(
            patience=self.patience, 
            compare=self.criterion.compare,
            metric=self.criterion.update_target
            )

        for epoch in range(self.n_epochs):
            self.model.train()
            for sample in tqdm.tqdm(self.data_loaders['train']):
                self.optimizer.zero_grad(set_to_none=True)
                
                targets = self.model.module.get_targets(sample)
                net_output  = self.model(**sample['net_input'])
                
                loss = self.criterion('loss', net_output, targets)
                loss['text_loss'].backward()
                self.optimizer.step()
                
                with torch.no_grad():
                    acc = self.criterion('acc', net_output, targets)

            with torch.no_grad():
                epoch_log = self.criterion.get_epoch_dict(len(self.data_loaders['train']))

            summary = utils.log_from_dict(epoch_log, 'train', epoch)
            if not self.config['debug']:
                wandb.log(summary)
            
            should_stop = self.validate(epoch)
            if should_stop:
                break

        self.test()
        if not self.config['debug']:
            wandb.finish(0)

    def inference(self, epoch, subsets):
        self.model.eval()
        generated, real = [], []

        with torch.no_grad():
            for subset in subsets:
                print(f"Inference on {subset}")

                for sample in tqdm.tqdm(self.data_loaders[subset]):
                    targets = self.model.module.get_targets(sample)

                    if self.config['sample'] and self.config['test_only']:
                        # Generation mode
                        net_output = self.model.module.generate(**sample['net_input'])

                        generated.append(net_output.detach().cpu().numpy())
                        real.append(targets['text_input_ids'].detach().cpu().numpy())

                        if len(generated) * self.config["per_gpu_batchsize"] >= self.config["num_samples"]:
                            # Construct and save output filenames
                            modality = ''.join(self.config['modality'])
                            test_sets = ''.join(self.config['test_sets'])

                            components = [
                                'gen', modality, test_sets, self.config["bert_type"],
                                self.config["topk"], self.config["temperature"],
                                self.config["prevent_too_short"], self.config["prevent_repeat_ngram"]
                            ]
                            
                            filename = '_'.join(list(map(str, components)))
                            os.makedirs("output", exist_ok=True)
                            gen_filename = f'output/{filename}'
                            real_filename = gen_filename.replace('gen_', 'real_')

                            np.save(gen_filename, np.vstack(generated).astype(np.int32))
                            np.save(real_filename, np.vstack(real).astype(np.int32))
                            return
                    else:
                        # Evaluation mode
                        net_output = self.model(**sample['net_input'])
                        loss = self.criterion('loss', net_output, targets)
                        acc = self.criterion('acc', net_output, targets)   
                
                epoch_log = self.criterion.get_epoch_dict(len(self.data_loaders[subset]))
                summary = utils.log_from_dict(epoch_log, subset, epoch)
                if not self.config['debug']:
                    wandb.log(summary)

        return epoch_log

    def validate(self, epoch):
        break_token = False
        epoch_log = self.inference(epoch, ['valid'])
        
        if self.early_stopping(epoch_log[self.criterion.update_target]):
            utils.model_save(self.path, self.model, self.optimizer, epoch)

        if self.early_stopping.early_stop:
            logger.info(f'Early stopped! All valid finished at {epoch} ...')
            break_token=True
        return break_token

    def test(self):
        epoch, self.model, _ = utils.model_load(self.path, self.model, self.optimizer)
        epoch_log = self.inference(epoch, self.config['test_sets'])
        return epoch_log

        
