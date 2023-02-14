import torch
import torch.nn as nn
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import time

class ApxSVD(torch.nn.Module):
    def __init__(self, 
                embeddings,
                n):
        super().__init__()
        if isinstance(embeddings,torch.nn.parameter.Parameter):
            embeddings=embeddings.detach().cpu()
        S,V,D = np.linalg.svd(embeddings,full_matrices=False)
        self.SV = nn.Parameter(torch.tensor(S[:,:n]*V[:n], dtype=torch.float32))
        self.D = nn.Parameter(torch.tensor(D[:n], dtype=torch.float32))

    def forward(self, idx):
        out=self.SV[idx]@self.D
        return out


class Indexing(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def forward(ctx, embeds, scores):
        ctx.save_for_backward(embeds, scores)
        # embeds.shape                                              # (L, W, C)
        s = scores.shape                                            # (L, B, W)
        idx = scores.max(-1)[1].T                                   # (B, L)
        idx = idx.reshape(-1)                                       # (B*L)
        out = embeds[list((torch.arange(s[0]).repeat(s[1]), idx))]  # (B*L, C)
        out = out.reshape(s[1],-1)                                  # (B, L*C)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        embeds, scores = ctx.saved_tensors
        grad_output=grad_output.view(-1, embeds.shape[0], embeds.shape[2])

        grad_embeds = grad_scores = None

        if ctx.needs_input_grad[0]:
            grad_embeds = torch.bmm(scores.permute(0,2,1), grad_output.permute(1,0,2))

        if ctx.needs_input_grad[1]:
            grad_scores = torch.bmm(grad_output.permute(1,0,2), embeds.permute(0,2,1))

        return grad_embeds, grad_scores


class CodeBook(torch.nn.Module):

    def __init__(self, 
        levels, 
        feature_dim,
        num_words,
        feature_std         : float        = 1.0,
        feature_bias        : float        = 0.0,
        codebook_bitwidth=8):

        super().__init__()

        self.levels=levels
        self.feature_dim=feature_dim
        self.num_words=num_words
        self.feature_std=feature_std
        self.feature_bias=feature_bias
        self.bitwidth=codebook_bitwidth

        self.dictionary_size = 2 ** self.bitwidth
        self.dictionary = nn.Parameter(torch.randn(self.levels, self.dictionary_size, self.feature_dim) * self.feature_std + self.feature_bias)
        self.feats=nn.Parameter(torch.randn(self.levels, self.num_words, self.dictionary_size))
        self.indexing_func=Indexing.apply
    
    def forward(self, idx):
        if self.feats is not None:                                                  # logits.shape (L,B,W)
            logits=self.feats[:,idx.long()] 
            return self.indexing_func(self.dictionary, logits)
        else:
            indices = self.fixed_indices[idx.long()]                             # indices.shape (B,L)
            indices = indices.reshape(-1).long()                                                 # (B*L)
            out = self.dictionary[list((torch.arange(self.levels).repeat(len(idx)), indices))]   # (B*L,C)
            out = out.reshape(len(idx),-1)                                                       # (B, L*C)
            return out
    
    def fix_indices(self):
        self.fixed_indices = nn.Parameter((self.feats.max(-1)[1]).to(torch.int8).T, requires_grad=False)
        self.feats = None


class ApproxEmbed(torch.nn.Module):
    def __init__(self, 
        levels, 
        feature_dim,
        num_words,
        output_dims,
        feature_std         : float        = 1.0,
        feature_bias        : float        = 0.0,
        codebook_bitwidth=8,
        neurons=64,
        nn_levels=2):
        
        super().__init__()
   
        self.B=CodeBook(levels, 
                        feature_dim,
                        num_words,
                        feature_std= feature_std,
                        feature_bias= feature_bias,
                        codebook_bitwidth=8)

        # REQUIRES TORCH 2.0 NIGHTLY ON LINUX
        #self.B=torch.compile(self.B,
        #            mode='max-autotune',
        #            fullgraph=True)
        
        self.N=torch.nn.Sequential()

        prec_dim = feature_dim*levels
        for i in range(nn_levels):
            self.N.append(torch.nn.Linear(prec_dim, neurons))
            self.N.append(torch.nn.LeakyReLU())
            prec_dim = neurons
        self.N.append(torch.nn.Linear(prec_dim, output_dims))

    def forward(self, idx):
        x = self.B(idx.reshape(-1))
        x = self.N(x).reshape(*idx.shape,-1)
        return x

    def fix_indices(self):
        self.B.fix_indices()




def train_apx(apx, embeddings, epochs=1000, batch_size=2**10, checkpoint_every=100, save_path='', optimizer=None, norm=2, verbose=False, device='cuda'):
    
    if optimizer is None:
        optimizer=torch.optim.Adam(apx.parameters(),lr=0.01)

    h=[]
    run={}
    run['level']=apx.B.levels
    run['channel']=apx.B.feature_dim
    best_loss=np.inf
    
    order=np.arange(embeddings.shape[0])
    for epoch in range(epochs):
        t0=time.time()
        tot_loss=0
        np.random.shuffle(order)
        for batch in range(int(np.ceil(len(order)/batch_size))):

            positions = torch.tensor(order[batch*batch_size:(batch+1)*batch_size],dtype=int,device=device)
            embed_preds = apx(positions)

            embeds = embeddings[positions]
            loss=torch.mean(torch.pow(torch.abs(embed_preds)-embeds, norm))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            tot_loss+=loss.detach().cpu()

        average_loss=tot_loss.numpy()/int(np.ceil(len(order)/batch_size))
        h.append(average_loss)
        if verbose:
            print(epoch, f'{average_loss:.5g}'.ljust(80), time.time()-t0, end='\r')
    
        if epoch%checkpoint_every==checkpoint_every-1:
            # save the results to be analized later
            run['loss']=h
            
            if tot_loss<best_loss:# save the best model
                best_loss=tot_loss
                run['apx']=apx.state_dict()
                torch.save(run, save_path + str(run['level']) + '_' + str(run['channel'])+'.pth')
    if verbose:
        print()
    apx.load_state_dict(torch.load(save_path + str(run['level']) + '_' + str(run['channel'])+'.pth')['apx'])
    run['apx']=apx
    return run

def calc_size(x):
   if isinstance(x,list) or isinstance(x,torch.nn.ParameterList):
      tot=np.sum([calc_size(elem) for elem in x])
   else:
      if isinstance(x,torch.Tensor):
         x=x.detach().cpu()
      x=np.array(x)
      tot=x.size * x.itemsize
   return tot


def load_runs(runs_folder):

    runs = []
    for name in os.listdir(runs_folder):
        run = load_run(runs_folder + name)
        runs.append(run)
    return runs

def load_run(filename):

    run=torch.load(filename)

    levels = run['level']
    channels = run['channel']
    nn_weights = [run['apx'][p] for p in run['apx'].keys() if '.weight' in p]
    nn_levels = len(nn_weights)-1
    neurons=len(nn_weights[0])
    embeddings_size=len(nn_weights[-1])
    feats=run['apx']['B.feats']
    if feats is not None:
        n_embeddings=feats.shape[1]
    else:
        n_embeddings=feats.shape[1]
    bits = int(math.log2(run['apx']['B.dictionary'].shape[1]))

    apx = ApproxEmbed(levels = levels, 
                feature_dim = channels,
                num_words = n_embeddings,
                output_dims = embeddings_size,
                feature_std = 0.1,
                feature_bias = 0.0,
                codebook_bitwidth=bits,
                neurons = neurons,
                nn_levels = nn_levels)

    if feats is None:
        apx.fix_indices()
    
    run['apx']=apx
    return run

def plot_compare_result_runs(runs, embeddings, bits=8):

    levs={}
    for run in runs:
        if run['level'] not in levs.keys():
            levs[run['level']]=[]
        levs[run['level']].append(run)

    LEVELS = list(levs.keys())
    max_s=torch.max(embeddings).item()

    color = plt.cm.viridis(1-np.linspace(0, 1, len(LEVELS)))

    fig, ax = plt.subplots(figsize=(15,10))
    for l in levs.keys():
        print(l)
        size= np.array([calc_size(list(run['apx'].parameters())) for run in levs[l]])/1e6
        loss = np.array([np.min(run['loss']) for run in levs[l]])
        idx = np.argsort(size)
        snr = 10*np.log10(max_s/loss)
        txt = [run['channel'] for run in levs[l]]
        ax.plot(size[idx], snr[idx], '.', color=color[np.where(np.array(LEVELS)==l)])
        ax.plot(size[idx], snr[idx], color=color[np.where(np.array(LEVELS)==l)])
        

    S,V,D = np.linalg.svd(embeddings.detach().cpu(),full_matrices=False)

    # plot 
    N=list(np.arange(40)+1)

    size=np.array([calc_size([S[:,:n],V[:n],D[:n]]) for n in N])/1e6
    loss=np.array([torch.mean(torch.square(torch.tensor((S[:,:n]*V[:n])@D[:n],device=embeddings.device)-embeddings)).item() for n in N])
    snr = 10*np.log10(max_s/loss)

    ax.plot(size, snr,'.', color='r')
    ax.plot(size, snr, color='r')

    plt.xlabel('Size MB')
    plt.ylabel('PSNR')