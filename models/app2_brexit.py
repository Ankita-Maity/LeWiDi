from transformers import AutoModel,AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math






class Aux(nn.Module):
    def __init__(self,aux_dim) -> None:
        super().__init__()
        self.linear1 = nn.Linear(6,aux_dim)
        # self.nonl = nn.LeakyReLU()
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
        B,N,H = q.shape

        q,k,v = q.reshape(B,N,8,H//8), k.reshape(B,N,8,H//8), v.reshape(B,N,8,H//8)
        q,k,v = q.permute(0,2,1,3),k.permute(0,2,1,3),v.permute(0,2,1,3)
        


        
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(self.aux_dim//8)  # scaled dot product
        

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v.permute(0,2,1,3).reshape(B,N,H)



class App2(nn.Module):
    def __init__(self,model,use_soft_loss=False,ax_dim=128,hidden_dim=256,dropout=0.2) -> None:
        super().__init__()
        self.aggro_mlp= Aux(ax_dim)
        self.aoff_mlp= Aux(ax_dim)
        self.LM = model
        self.aprojector = nn.Linear(768,2*ax_dim)
        self.bprojector = nn.Linear(2*ax_dim,6)
        self.amixer1=Mixer(ax_dim)
        self.ln1=nn.LayerNorm(2*ax_dim)
        self.amixer2=Mixer(ax_dim)
        self.ln2=nn.LayerNorm(2*ax_dim)
        self.amixer3=Mixer(ax_dim)
        self.ln3=nn.LayerNorm(2*ax_dim)

        self.affn1 = nn.Sequential(*[nn.Linear(2*ax_dim,hidden_dim),
                                   nn.LeakyReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(hidden_dim,hidden_dim),
                                   ])
        
        self.affn2 = nn.Sequential(*[nn.Linear(2*ax_dim,hidden_dim),
                                   nn.LeakyReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(hidden_dim,hidden_dim),
                                   ])

        self.affn3 = nn.Sequential(*[nn.Linear(2*ax_dim,hidden_dim),
                                   nn.LeakyReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(hidden_dim,hidden_dim),
                                   ])


    def forward(self,X,aggro,off,y):
        emb1 =self.LM(X)['pooler_output']
        emb1 = self.aprojector(emb1)
        emb2 = self.aggro_mlp(aggro)
        emb3 = self.aoff_mlp(off)

        aux = torch.cat([emb2,emb3],dim=1)
        stacked_emb = torch.stack([emb1,aux],dim=1)
        
        final_emb = self.amixer1(stacked_emb)+stacked_emb
        final_emb = F.layer_norm(final_emb,final_emb.shape)
        final_emb = self.affn1(final_emb) + final_emb
        final_emb = F.layer_norm(final_emb,final_emb.shape)

        final_emb = self.amixer2(final_emb)+final_emb
        final_emb = F.layer_norm(final_emb,final_emb.shape)
        final_emb = self.affn2(final_emb) + final_emb
        final_emb = F.layer_norm(final_emb,final_emb.shape)

        final_emb = self.amixer3(final_emb)+final_emb
        final_emb = F.layer_norm(final_emb,final_emb.shape)
        final_emb = self.affn3(final_emb) + final_emb
        final_emb = F.layer_norm(final_emb,final_emb.shape)


        logits = self.bprojector(final_emb[:,:-1,:])
        return logits.squeeze(dim=1)



if __name__=="__main__":
    X = torch.randint(1,400,(16,80))
    l1 = torch.rand((16,6))
    l2 = torch.rand((16,6))
    y = torch.empty(16,6).random_(2)
    loss_fn = nn.BCEWithLogitsLoss()
    LM = AutoModel.from_pretrained('vinai/bertweet-base')
    m = App2()
    o = m(X,l1,l2,y)
    print(o)
    print(y)
    loss = loss_fn(o,y)

    print(loss)

        


