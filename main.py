import os, glob, sys, random, pickle, copy, resource, logging
import numpy as np
import torch
import torch.multiprocessing as mp
from mmtg.config import ex
from mmtg.datamodules.dataset import mmtg_data_loader
from mmtg.modules import MMTGTransformer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s %(name)s %(message)s)))",
    datefmt="%Y-%m-%d %H:%M:%S",
    level = os.environ.get("LOGLEVEL", "INFO").upper(),
    stream = sys.stdout
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    mp.set_sharing_strategy('file_system')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    

@ex.automain
def main(_config):
    #Seed pivotting
    set_seed(_config['seed'])

    #Data modules
    data_loaders = mmtg_data_loader(_config)
    
    #Module
    model = MMTGTransformer(_config)
    
    #Trainer
    from trainer import Trainer
    trainer = Trainer(model, data_loaders, _config)

    if not _config["test_only"]:
        trainer.train()
    else:
        trainer.test()
    