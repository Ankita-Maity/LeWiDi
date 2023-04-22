import torch

import numpy as np
from tqdm import tqdm
from utils.utils import *
import numpy as np
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)



class Trainer():
    def __init__(self,model,train_loader,loss_fn,optimizer,epochs=1,
                 use_soft_loss=False,
                scheduler=None,val_loader=None,
                device=torch.device('cpu'),logger=None,) -> None:
        self.model=model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer=optimizer
        self.device=device
        self.model.to(device)
        self.loss_fn = loss_fn
        self.num_epoch = epochs
        self.scheduler=scheduler
        self.wandb=logger
        self.use_soft_loss = use_soft_loss
        self.mode="hard" if not use_soft_loss else "soft"

    def training_step(self,batch,batch_idx):
        self.optimizer.zero_grad()

        batch = [i.to(self.device) for i in batch]
        
        otp = self.model(*batch)
       
        loss = self.loss_fn(otp,batch[-1])
        loss.backward()
        self.optimizer.step()
        
        self.wandb.log({"Training Loss":loss.item()})
        return otp,loss.item()

    def validation_step(self,batch,batch_idx):
        batch = [i.to(self.device) for i in batch]
        
        with torch.no_grad():
            otp = self.model(*batch)
            loss = self.loss_fn(otp,batch[-1])
            self.wandb.log({"Validation Loss":loss.item()})
        return otp,loss.item()

    def predict_step(self,batch,batch_idx):
        batch = [i.to(self.device) for i in batch]
        
        with torch.no_grad():
            otp = self.model(*batch)
        return otp

    def fit(self,metrics_callbacks=None):
        for epoch in range(1,self.num_epoch+1):
            self.model.train()
            train_losses=[]
            targets,outputs =[],[]
            for batch_idx,batch in tqdm(enumerate(self.train_loader)):
                
                otp,l = self.training_step(batch[:-1],batch_idx)
                train_losses.append(l)
                outputs.append(otp.detach().cpu().numpy())
                targets.append(batch[-2].cpu().numpy())
            print('Epoch ',epoch,' Training Loss is ',sum(train_losses)/len(self.train_loader))
            self.wandb.log({"Train Epoch Loss":sum(train_losses)/len(self.train_loader)})
            print('Training')
            report(np.vstack(targets),np.vstack(outputs),self.mode)
            self.wandb.log({"Train targets":pct_preds(np.vstack(outputs),mode=self.mode)})

            if metrics_callbacks is not None:
                for metric_name,metric_func in metrics_callbacks.items():
                    self.wandb.log({metric_name+" Training ":metric_func(np.vstack(targets),np.vstack(outputs),mode=self.mode)})

            if self.val_loader is not None:
                self.model.eval()
                val_losses=[]
                targets,outputs =[],[]
                for batch_idx,batch in tqdm(enumerate(self.val_loader)):
                    otp,l = self.validation_step(batch[:-1],batch_idx)
                    val_losses.append(l)
                    outputs.append(otp.detach().cpu().numpy())
                    targets.append(batch[-2].cpu().numpy())
                print('Epoch ',epoch,' Validation Loss is ',sum(val_losses)/len(self.val_loader))
                self.wandb.log({"Validation Epoch Loss":sum(val_losses)/len(self.val_loader)})
                self.wandb.log({"Validation preds":pct_preds(np.vstack(outputs),mode=self.mode)})
                print('Validation')
                report(np.vstack(targets),np.vstack(outputs),self.mode)
                
                if metrics_callbacks is not None:
                    for metric_name,metric_func in metrics_callbacks.items():
                        self.wandb.log({metric_name+" Validation ":metric_func(np.vstack(targets),np.vstack(outputs),mode=self.mode)})
                
            if self.scheduler is not None:
                self.scheduler.step()

    
    def predict(self,test_loader,filename="test_preds",writer=None):
        otps=[]
        indexes=[]
        for batch_idx,batch in tqdm(enumerate(test_loader)):
            otp = self.predict_step(batch[:-1],batch_idx)
            otps.append(otp.detach().cpu().numpy())
            indexes.append(batch[-1].cpu().numpy())

        outputs=np.vstack(otps)
        # import pdb;pdb.set_trace()
        indexes = np.concatenate(indexes).reshape(-1,1)
        
        if writer is not None:
            writer(np.hstack([indexes,outputs]),filename)
       
        return np.hstack([indexes,outputs])
            
        





