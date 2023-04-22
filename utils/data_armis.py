import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
from utils.utils import *
import nltk






df_train = pd.read_json('../data_post-competition/ArMIS_dataset/ArMIS_train.json').T
df_val= pd.read_json('../data_post-competition/ArMIS_dataset/ArMIS_dev.json').T
df_test = pd.read_json('../data_post-competition/ArMIS_dataset/ArMIS_test.json').T


def preprocess(df,split='train'):
    if split!='test':
        df[['lab'+str(i) for i in range(3)]]= df['annotations'].str.split(',',expand=True) 

    df['prob'] = df['soft_label'].apply(pd.Series)['1']
    df = df.drop(['annotation task','number of annotations','annotations','annotators','lang','soft_label','split','other_info'],axis=1)
    if split!='test':
        X= df.drop(['hard_label',	'lab0',	'lab1',	'lab2', 'prob'],axis=1)
        meta = list(range(df.shape[0]))
        y = df[['lab0',	'lab1',	'lab2']].values.astype(int)
    else:
        X= df.drop([	'hard_label', 'prob'],axis=1)
        meta = list(range(df.shape[0]))
        y = [[0]*3]*df.shape[0]
    return X,y,meta

X_train,y_train,meta_train = preprocess(df_train)
X_val,y_val,meta_val = preprocess(df_val,'val')
X_test,y_test,meta_test = preprocess(df_test,'test')

class dataset(Dataset):
    def __init__(self,df,labels,meta):
        super().__init__()
        self.df = df
        self.meta = meta
        self.labels = torch.Tensor(labels)

    def __getitem__(self,idx):
        texts = self.df.iloc[idx]['text']
        
        return {"data":[texts,self.labels[idx]],"meta":self.meta[idx]}
    
    def __len__(self):
        return len(self.labels)

def collate_fn(tokenizer,pad_token,use_soft_loss,device,batch):
    def text_pipeline(texts):
        tokens = tokenizer(texts)['input_ids']
        return tokens
    def pad_sequence(text_list):
        max_len = max([len(i) for i in text_list])
        return [i+[pad_token]*(max_len-len(i)) for i in text_list]


    text_list = []  
    label_list= []
    meta_list=[]
    
    for i in batch:
        (_text,_lab)=i['data']
        _meta = i['meta']
        processed_text = text_pipeline(_text)
        text_list.append(processed_text)

        if use_soft_loss:
            label_list.append(torch.Tensor([1-_lab.mean(),_lab.mean()]))
        else:
            label_list.append(_lab)
        meta_list.append(_meta)
    
    label_list = torch.stack(label_list,dim=0)
    meta_list = torch.LongTensor(meta_list)
    text_list = torch.LongTensor(pad_sequence(text_list))
    
    return text_list,label_list,meta_list

train_ds = dataset(X_train,y_train,meta_train)
val_ds = dataset(X_val,y_val,meta_val)
test_ds = dataset(X_test,y_test,meta_test)
