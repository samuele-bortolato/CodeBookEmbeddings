import torch
import torch.nn as nn
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import time
import random 

def set_random_seed(random_seed: int = 42) -> None:
    """Set the random seed for reproducibility. The seed is set for the random library, the numpy library and the pytorch 
    library. Moreover the environment variable `TF_DETERMINISTIC_OPS` is set as "1".

    Parameters
    ----------
    random_seed : int, optional
        The random seed to use for reproducibility (default 42).
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.environ['TF_DETERMINISTIC_OPS'] = '1'

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
        # embeds.shape                                              # (L, W, C)
        s = scores.shape                                            # (L, B, W)
        idx = scores.max(-1)[1].T                                   # (B, L)
        idx = idx.reshape(-1)                                       # (B*L)
        out = embeds[list((torch.arange(s[0]).repeat(s[1]), idx))]  # (B*L, C)
        out = out.reshape(s[1],-1)                                  # (B, L*C)
        ctx.save_for_backward(embeds, idx)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        embeds, idx = ctx.saved_tensors
        grad_output=grad_output.view(-1, embeds.shape[0], embeds.shape[2]) #(B,L,C)

        grad_embeds = grad_scores = None

        if ctx.needs_input_grad[0]:
            grad_embeds = torch.zeros(embeds.shape[1],
                                      embeds.shape[0],
                                      embeds.shape[2],
                                      device=grad_output.device, 
                                      dtype=grad_output.dtype)      # (W, L, C)
            idx = idx.view(-1,embeds.shape[0],1)                    # (B, L, 1)
            grad_embeds.scatter_add_(0,idx,grad_output)
            grad_embeds=grad_embeds.permute(1,0,2)                  # (L, W, C)

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
        if self.feats is not None:
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

class CodeBook2(torch.nn.Module):

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
        self.bitwidth=codebook_bitwidth//2

        self.dictionary_size = 2 ** self.bitwidth
        self.dictionary = nn.Parameter(torch.randn(self.levels, self.dictionary_size**2, self.feature_dim) * self.feature_std + self.feature_bias)
        self.feats0=nn.Parameter(torch.randn(self.levels, self.num_words, self.dictionary_size))
        self.feats1=nn.Parameter(torch.randn(self.levels, self.num_words, self.dictionary_size))
        self.indexing_func=Indexing.apply
    
    def forward(self, idx):
        f0=self.feats0[:,idx.long()][...,None]
        f1=self.feats1[:,idx.long()][...,None,:]
        S=f0.shape
        logits=(f0*f1).reshape(S[0],S[1],-1)
        return self.indexing_func(self.dictionary, logits)

    def fix_indices(self):
        if self.feats0 is not None:
            f0=self.feats0[...,None]
            f1=self.feats1[...,None,:]
            S=f0.shape
            self.fixed_indices = nn.Parameter(((f0*f1).reshape(S[0],S[1],-1).max(-1)[1]).to(torch.int8).T, requires_grad=False)
            self.feats0 = None
            self.feats1 = None


class ApproxEmbed2(torch.nn.Module):
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
   
        self.B=CodeBook2(levels, 
                        feature_dim,
                        num_words,
                        feature_std= feature_std,
                        feature_bias= feature_bias,
                        codebook_bitwidth=codebook_bitwidth)

        # self.B=torch.compile(self.B,
        #             mode='max-autotune',
        #             fullgraph=True)
        
        self.N=torch.nn.Sequential()

        prec_dim = feature_dim*levels
        for i in range(nn_levels):
            self.N.append(torch.nn.Linear(prec_dim, neurons))
            self.N.append(torch.nn.LeakyReLU())
            prec_dim = neurons
        self.N.append(torch.nn.Linear(prec_dim, output_dims))

    def forward(self, idx):
        x = self.B(idx)
        x = self.N(x)
        return x

    def fix_indices(self):
        self.B.fix_indices()




def train_apx(apx, embeddings, epochs=1000, batch_size=2**10, checkpoint_every=100, lr=0.01, save_path='', optimizer=None, loss_function=torch.nn.functional.mse_loss, verbose=False, device='cuda', name=''):
    
    if optimizer is None:
        if hasattr(apx.B,'feats'):
            optimizer=torch.optim.AdamW([
                            {'params': apx.B.feats,'weight_decay':1},
                            {'params': [parameter for parameter in apx.parameters() if parameter is not apx.B.feats]}
                        ], lr=lr, weight_decay=0)
        elif (hasattr(apx.B,'feats0')and hasattr(apx.B,'feats1')):
            optimizer=torch.optim.AdamW([
                            {'params': [apx.B.feats0,apx.B.feats1],'weight_decay':1},
                            {'params': [parameter for parameter in apx.parameters() if (parameter is not apx.B.feats0) and (parameter is not apx.B.feats1)]}
                        ], lr=lr, weight_decay=0)
        else:
            optimizer=torch.optim.Adam(apx.parameters(), lr=lr, weight_decay=0)

    name = str(apx.B.levels) + '_' + str(apx.B.feature_dim) + name + '.pth'

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
            loss=loss_function(embed_preds,embeds)

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
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(run, save_path + name)
    if verbose:
        print()
    apx.load_state_dict(torch.load(save_path + name)['apx'])
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


def load_runs(runs_folder, fixed=False):

    runs = []
    for name in os.listdir(runs_folder):
        run = load_run(runs_folder + name, fixed)
        runs.append(run)
    return runs

def load_run(filename, fixed=False):

    run=torch.load(filename)

    levels = run['level']
    channels = run['channel']
    nn_weights = [run['apx'][p] for p in run['apx'].keys() if '.weight' in p]
    nn_levels = len(nn_weights)-1
    neurons=len(nn_weights[0])
    embeddings_size=len(nn_weights[-1])
    bits = int(math.log2(run['apx']['B.dictionary'].shape[1]))
    if 'B.feats' in run['apx']:
        feats=run['apx']['B.feats']
        if feats is not None:
            n_embeddings=feats.shape[1]
        else:
            n_embeddings=run['apx']['B.fixed_indices'].shape[1]

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
        apx.load_state_dict(run['apx'])
        if fixed:
            apx.fix_indices()
    else:
        feats0=run['apx']['B.feats0']
        if feats0 is not None:
            n_embeddings=feats0.shape[1]
        else:
            n_embeddings=run['apx']['B.fixed_indices'].shape[1]
        
        apx = ApproxEmbed2(levels = levels, 
                    feature_dim = channels,
                    num_words = n_embeddings,
                    output_dims = embeddings_size,
                    feature_std = 0.1,
                    feature_bias = 0.0,
                    codebook_bitwidth=bits,
                    neurons = neurons,
                    nn_levels = nn_levels)
        if feats0 is None:
            apx.fix_indices()
        apx.load_state_dict(run['apx'])
        if fixed:
            apx.fix_indices()
        
    
    run['apx']=apx
    return run

def plot_compare_result_runs(runs, embeddings, bits=8):

    levs={}
    for run in runs:
        if run['level'] not in levs.keys():
            levs[run['level']]=[]
        levs[run['level']].append(run)

    LEVELS = np.sort(np.array(list(levs.keys())))
    max_s=torch.max(embeddings).item()

    color = plt.cm.viridis(1-np.linspace(0, 1, len(LEVELS)))

    fig, ax = plt.subplots(figsize=(7,5))
    for l in np.sort(np.array(list(levs.keys())),axis=0):
        print(l)
        size= np.array([calc_size(list(run['apx'].parameters())) for run in levs[l]])/1e6
        loss = np.array([np.min(run['loss']) for run in levs[l]])
        idx = np.argsort(size)
        #snr = 10*np.log10(max_s/loss)
        txt = [run['channel'] for run in levs[l]]
        ax.plot(size[idx], loss[idx], '.', color=color[np.where(np.array(LEVELS)==l)])
        ax.plot(size[idx], loss[idx], color=color[np.where(np.array(LEVELS)==l)], label=f"{l} levels")
        

    S,V,D = np.linalg.svd(embeddings.detach().cpu(),full_matrices=False)

    # plot 
    N=list(np.arange(40)+1)

    size=np.array([calc_size([S[:,:n],V[:n],D[:n]]) for n in N])/1e6
    loss=np.array([torch.mean(torch.square(torch.tensor((S[:,:n]*V[:n])@D[:n],device=embeddings.device)-embeddings)).item() for n in N])
    #snr = 10*np.log10(max_s/loss)

    ax.plot(size, loss,'.', color='r')
    ax.plot(size, loss, color='r', label=f"SVD")

    plt.xlabel('Size MB')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.legend()