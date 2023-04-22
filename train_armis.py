import torch
import torch.nn as nn
import torch.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
import os
from utils.utils import *
from utils.data_armis import *
from utils.trainer import Trainer

from models.app2_armis import App2
import nltk


from transformers import AutoModel,AutoTokenizer
import torch


device=torch.device('cuda:0')
LM = AutoModel.from_pretrained('aubmindlab/bert-base-arabert')
tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabert')

for param in LM.parameters():
    param.requires_grad = False


import wandb

wandb.init(project="armis")
use_soft_loss=False

def collate_wrapper(func):
    def wrapper(*args,**kwargs):
        pad_token=1
        return func(tokenizer,pad_token,use_soft_loss,device,*args,**kwargs)
    return wrapper


wrapped_collate_fn = collate_wrapper(collate_fn)

    



train_loader = DataLoader(train_ds,batch_size=16,shuffle=True,collate_fn = wrapped_collate_fn,num_workers=10)
val_loader = DataLoader(val_ds,batch_size=16,collate_fn = wrapped_collate_fn,num_workers=10)
test_loader = DataLoader(test_ds,batch_size=16,collate_fn = wrapped_collate_fn,num_workers=10)


print('Train data size ',len(train_ds),' Number of steps in epoch ',len(train_ds)/len(train_loader))
print('Validation data size ',len(val_ds))
print('Test data size ',len(test_ds))

bceloss = nn.BCEWithLogitsLoss()
nll=nn.NLLLoss()

model = App2(LM,use_soft_loss)

optimizer,scheduler = get_config_optimizer(model)

wandb.watch(model,log_freq=50, log="all")
trainer = Trainer(model,train_loader=train_loader,
            loss_fn=bceloss,
            optimizer=optimizer,
            epochs = 30,
            scheduler=scheduler,
            val_loader=val_loader,
            device=device,
            logger=wandb,
            use_soft_loss=use_soft_loss
            )

trainer.fit({"ce":bceloss_cpu,"f1":f1_metric,"p":p_metric,"r":r_metric})

if not os.path.exists('predictions/'):
    os.makedirs('predictions/')


trainer.predict(train_loader,'predictions/train_armis',csv_writer)
trainer.predict(val_loader,'predictions/val_armis',csv_writer)
trainer.predict(test_loader,'predictions/test_armis',csv_writer)
