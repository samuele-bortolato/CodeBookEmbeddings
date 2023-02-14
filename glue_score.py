import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoModelForSequenceClassification
import torchmetrics
import datasets
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

GLUE_TASKS={
    'cola':{'type':'classification', 'metrics':['Matthews']},
    'sst2':{'type':'classification', 'metrics':['Accuracy']},
    'mrpc':{'type':'classification', 'metrics':['F1','Accuracy']},
    'stsb':{'type':'regression', 'final_activation':'sigmoid', 'max':5, 'metrics':['Pearson','Spearman']},
    'qqp':{'type':'classification', 'metrics':['F1']},
    'mnli':{'type':'classification', 'num_classes':3, 'val_datasets':['mnli_mismatched','mnli_matched'], 'metrics':['Accuracy']},
    'qnli':{'type':'classification', 'metrics':['Accuracy']},
    'rte':{'type':'classification', 'metrics':['Accuracy']},
    'wnli':{'type':'classification', 'metrics':['Accuracy']}
}

def make_model(model_name, args, device='cuda'):
    if args['type']=='classification':
        if 'num_classes' in args.keys():
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=args['num_classes'], ignore_mismatched_sizes=True)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
    elif args['type']=='regression':
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, ignore_mismatched_sizes=True)
    
    return model.to(device)


def validate(model, tokenizer,val_dataloader, metrics, args, device='cuda'):
    model.eval()
    with torch.no_grad():
        
        results=[]
        for dataloader in val_dataloader:

            predictions=[]
            targets=[]

            for val_batch in dataloader:

                val_inputs=tokenizer(*[val_batch[k] for k in val_batch.keys() if (k!='label')and(k!='idx')], truncation=True, padding='max_length', max_length=128, return_tensors='pt').to(device)
                outputs = model(input_ids=val_inputs['input_ids'], attention_mask=val_inputs['attention_mask'].to(torch.int32))[0]
                
                if args['type']=='classification':
                    preds = outputs.argmax(-1)
                elif args['type']=='regression':
                    if args['final_activation']=='sigmoid':
                        preds = torch.sigmoid(outputs[:,0])*(args['max']-args['min'])+args['min']
                    else:
                        preds = outputs[:,0]

                predictions.append(preds)
                targets.append(val_batch["label"])
            
            predictions=torch.concat(predictions,0).to(device)
            targets=torch.concat(targets,0).to(device).to(torch.float32)

            results+=[metric(predictions,targets).detach().cpu() for metric in metrics]
    
    model.train()
    return results

def get_dataloaders(args, task_name, batch_size):

    train_dataset = datasets.load_dataset("glue", task_name, split="train")
    if 'val_datasets' in args.keys():
        val_dataset = [datasets.load_dataset("glue", val_name, split="validation") for val_name in args['val_datasets']]
    else:
        val_dataset = [datasets.load_dataset("glue", task_name, split="validation")]

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = [DataLoader(v_d, batch_size=batch_size, shuffle=False) for v_d in val_dataset]

    return train_dataloader,val_dataloader

def get_metrics(args, device='cuda'):
    if 'num_classes' not in args.keys():
        args['num_classes']=2
    metrics = []
    for metric in args['metrics']:
        if metric=='Accuracy':
            metrics.append(torchmetrics.classification.MulticlassAccuracy(args['num_classes']).to(device))
        elif metric=='F1':
            metrics.append(torchmetrics.classification.MulticlassF1Score(args['num_classes']).to(device))
        elif metric=='Matthews':
            metrics.append(torchmetrics.classification.MulticlassMatthewsCorrCoef(args['num_classes']).to(device))
        elif metric=='Pearson':
            metrics.append(torchmetrics.PearsonCorrCoef().to(device))
        elif metric=='Spearman':
            metrics.append(torchmetrics.SpearmanCorrCoef().to(device))
    return metrics

def Glue(model, tokenizer, task_name, args, epochs=10, lr=1e-5, batch_size=32, steps_validate=0.25, device='cuda'):

    train_dataloader,val_dataloader = get_dataloaders(args, task_name, batch_size)

    metrics = get_metrics(args)

    if args['type']=='regression':
        if 'min' not in args.keys():
            args['min']=0
        if 'min' not in args.keys():
            args['max']=1

    model.train()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    history=[]
    val_history=[]

    for epoch in range(epochs):
        tot_loss=0
        tot_elems=0
        progress = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch: {epoch+1}')
        for batch_idx, batch in progress:

            optimizer.zero_grad()

            train_inputs = tokenizer(*[batch[k] for k in batch.keys() if (k!='label')and(k!='idx')], truncation=True, padding='max_length', max_length=128, return_tensors='pt').to(device)
            outputs = model(input_ids=train_inputs['input_ids'], attention_mask=train_inputs['attention_mask'].to(torch.int32))
            if args['type']=='classification':
                loss = torch.nn.functional.cross_entropy(outputs[0],batch['label'].to(device))
            elif args['type']=='regression':
                if args['final_activation']=='sigmoid':
                    preds=torch.sigmoid(outputs[0][:,0])*(args['max']-args['min'])+args['min']
                else:
                    preds = outputs[0][:,0]
                loss = torch.nn.functional.mse_loss(preds,batch['label'].to(device).to(torch.float32))
            else:
                raise Exception("unknown task type")

            loss.backward()
            optimizer.step()

            if steps_validate is not None:
                if isinstance(steps_validate, float):
                    steps_validate=int(steps_validate*len(train_dataloader))
                if batch_idx % steps_validate == steps_validate-1:
                    val_metrics = validate(model, tokenizer, val_dataloader, metrics, args)
                    val_history.append((len(history),val_metrics))

            tot_loss+=loss.detach().cpu()
            tot_elems+=len(batch)
            history.append(loss.detach().cpu())
            progress.set_postfix_str(f"loss: {tot_loss/tot_elems:.3g}")

    if 'val_datasets' in args.keys():
        return  {'batch_per_epoch':len(train_dataloader),'history':history} , {'metrics': [ v_d + " " + m for m in args['metrics'] for v_d in args['val_datasets']] , 'val_history':val_history}
    else:
        return {'batch_per_epoch':len(train_dataloader),'history':history} , {'metrics': args['metrics'], 'val_history':val_history}


def plot(H, V=None):
    fig, ax1 = plt.subplots()
    color = 'b'
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    h=H['history']
    n=len(h)//10
    ax1.plot(np.arange(len(h))/H['batch_per_epoch'],h, color=color, alpha=0.2)
    h1= np.convolve(h, np.ones(n)/n, mode='same')/(np.convolve(np.ones(len(h)), np.ones(n)/n, mode='same')+1e-7)
    ax1.plot((np.arange(len(h1))+len(h)-len(h1))/H['batch_per_epoch'],h1, color=color)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    if V is not None:
        VH=V['val_history']
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('metrics')
        colors = plt.cm.autumn(np.linspace(0, 1, len(V['metrics'])+1))
        for i, name in enumerate(V['metrics']):
            ax2.plot([x/H['batch_per_epoch'] for x,y in VH],[y[i] for x,y in VH],".", color=colors[i])

        ax2.legend([*V['metrics']])

    #fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def plot_results(run_dict):
    styles=['--',':']

    for task in run_dict.keys():
        fig, ax = plt.subplots(figsize=(15,5))
        fig.suptitle(task, fontsize=16)
        runs = run_dict[task]
        colors = plt.cm.plasma(np.linspace(0, 1, len(runs.keys())+1))
        for i,run in enumerate(runs.keys()):
            plt.subplot(1,2,1)
            plt.title('training loss')
            H=runs[run]['history']
            h=H['history']
            n=len(h)//10
            plt.plot(np.arange(len(h))/H['batch_per_epoch'],h, color=colors[i], alpha=0.3)
            h1= np.convolve(h, np.ones(n)/n, mode='same')/(np.convolve(np.ones(len(h)), np.ones(n)/n, mode='same')+1e-7)
            plt.plot((np.arange(len(h1))+len(h)-len(h1))/H['batch_per_epoch'],h1, color=colors[i], label=run)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.legend()
            
            plt.subplot(1,2,2)
            V=runs[run]['validation']
            if V is not None:
                VH=V['val_history']
                plt.title('metrics')
                for j, name in enumerate(V['metrics']):
                    if (i==0):
                        plt.plot([x/H['batch_per_epoch'] for x,y in VH],[y[j] for x,y in VH],styles[j], color=colors[i], label=name, alpha=0.7)
                    else:
                        plt.plot([x/H['batch_per_epoch'] for x,y in VH],[y[j] for x,y in VH],styles[j], color=colors[i], alpha=0.7)
            plt.legend([*V['metrics']])