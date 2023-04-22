from nltk.tokenize import TweetTokenizer
from emoji import demojize
import torch.nn as nn
from scipy.special import expit

import torch
from sklearn.metrics import f1_score,recall_score,precision_score,classification_report
import numpy as np
import pandas as pd

tweettokenizer = TweetTokenizer()


def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet):
    tokens = tweettokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())

def bceloss_cpu(targets_soft, predictions_soft,mode="hard", epsilon = 1e-12):                                
    predictions = np.clip(predictions_soft, epsilon, 1. - epsilon)                                      
    N = predictions.shape[0]*predictions.shape[1]
    ce = -np.sum(targets_soft*np.log(predictions+1e-9))/N
    return ce

def manipulate_labels(x,y,mode='hard'):
    if mode=='hard':
        return x, (expit(y)>0.5)*1
    else:
        return (x[:,0]<x[:,1])*1, (y[:,0]<y[:,1])*1

def f1_metric(targets, prediction,mode='hard'):
    targets,prediction = manipulate_labels(targets,prediction,mode)
    f1_wa = f1_score(targets, (expit(prediction)>0.5)*1, average = 'micro')              
    return f1_wa
def p_metric(targets, prediction,mode='hard'):
    targets,prediction = manipulate_labels(targets,prediction,mode)
    p = precision_score(targets, (expit(prediction)>0.5)*1, average = 'micro')             
    return p
def r_metric(targets, prediction,mode='hard'):
    targets,prediction = manipulate_labels(targets,prediction,mode)
    r = recall_score(targets, (expit(prediction)>0.5)*1, average = 'micro')               
    return r

def report(targets, prediction,mode='hard'):

    targets,prediction = manipulate_labels(targets,prediction,mode)

    rpt= classification_report(targets,prediction,zero_division=0)
    print(rpt)   

def pct_preds(predictions,mode='hard'):
    if mode=='hard':
        preds = (expit(predictions)>0.5)*1
        N = predictions.shape[0]*predictions.shape[1]
        return np.sum(preds)/N
    else:
        preds= (predictions[:,0]<predictions[:,1])*1
        print(preds)
        return np.mean(preds)

def get_config_optimizer(model,):
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-6)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,1e-6,1e-3,step_size_up=100,step_size_down=60,mode='triangular2',cycle_momentum = False)
    return optimizer,scheduler


def csv_writer(obj,filename):
    pd.DataFrame(obj).to_csv(filename+'.csv',index=False)
