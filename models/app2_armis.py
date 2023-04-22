from transformers import AutoModel,AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class Aux(nn.Module):
    def __init__(self,aux_dim) -> None:
        super().__init__()
        self.linear1 = nn.Linear(3,aux_dim)
        #self.nonl = nn.LeakyReLU()
        self.linear2 = nn.Linear(aux_dim,aux_dim)


    def forward(self,X):
        ans =  self.linear1(X)
        ans =  F.gelu(ans)
        ans = self.linear2(ans)


        return ans

class Mixer(nn.Module):
    def __init__(self,aux_dim) -> None:
        super().__init__()
        self.aux_dim =aux_dim
        self.Wq = nn.Linear(2*aux_dim,2*aux_dim)
        self.Wk = nn.Linear(2*aux_dim,2*aux_dim)
        self.Wv = nn.Linear(2*aux_dim,2*aux_dim)
        self.softmax=nn.Softmax(dim=-1)
    
    def forward(self,X,mask=None):

        q,k,v = self.Wq(X),self.Wk(X),self.Wv(X)

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 1)  # transpose
        score = (q @ k_t) / math.sqrt(self.aux_dim)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score



class App2(nn.Module):
    def __init__(self,model,use_soft_loss=False,ax_dim=256,hidden_dim=256,dropout=0) -> None:
        super().__init__()
        self.use_soft_loss =use_soft_loss
        self.LM = model
        self.aprojector = nn.Linear(768,2*ax_dim)

        num_classes =2 if use_soft_loss else 3

        self.affn = nn.Sequential(*[nn.Linear(2*ax_dim,hidden_dim),
                                   nn.LeakyReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(hidden_dim,10),
                                   nn.LeakyReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(10,num_classes)])


    def forward(self,X,y):
        emb1 =self.LM(X)['pooler_output']
        emb1 = self.aprojector(emb1)
        logits = self.affn(emb1)
        return logits



if __name__=="__main__":
    X = torch.randint(1,400,(16,80))
    l1 = torch.rand((16,6))
    l2 = torch.rand((16,6))
    y = torch.empty(16,6).random_(2)
    loss_fn = nn.BCEWithLogitsLoss()

    m = App2()
    o = m(X,l1,l2,y)
    print(o)
    print(y)
    loss = loss_fn(o,y)

    print(loss)

        


