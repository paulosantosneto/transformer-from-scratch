import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from models import *
from tqdm import tqdm
from torchinfo import summary

def train(model, optimizer, loss_fn, train_loader, test_loader, epochs, device):
    
    loss_h = []

    for epoch in tqdm(range(epochs), desc='Epochs'):
        
        batch_bar = tqdm(enumerate(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
        epoch_loss = 0
        
        for batch, (source, target, output) in batch_bar:
            source = source.to(device)
            target = target.to(device)
            output = output.to(device)

            logits = model(source, target)

            flattened_output = output.view(-1)
            batch_loss = loss_fn(logits.view(-1, logits.size()[-1]), flattened_output)
            
            epoch_loss += batch_loss.item()
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            batch_bar.set_postfix({'Loss': batch_loss.item()}, refresh=True)
        
        loss_h.append(epoch_loss)
        
    return loss_h
            
if __name__ == '__main__':
    
    args = get_args()
    
    if args.mode == 'train':

        # --- DataLoader ---
    
        train_dataset, test_dataset, en_vocab, pt_vocab, eng, pt = load_and_preprocessing_data(args)

        # --- Model ---
        
        model = Transformer(n=args.N, 
                            dm=args.dm, 
                            dff=args.dff, 
                            heads=args.heads,
                            source_vocab=en_vocab,
                            target_vocab=pt_vocab,
                            device=args.device
        ).to(args.device)
        
        # --- Model Architecture & Summary ---
        
        if args.verbose:
            for (source, target, _) in train_dataset:
                source = source.to(args.device)
                target = target.to(args.device)

                summary(model, [source.size(), target.size()], dtypes=[torch.long, torch.long], verbose=1)
                break
        
        # --- Optimizer & Loss Function ---
        
        #TODO scheduler and Adam with weight decay and LLRD
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-03, betas=(0.9, 0.98), eps=1e-09)
        
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0)

        # --- Train Model ---
        
        loss = train(model, optimizer, loss_fn, train_dataset, test_dataset, args.epochs, args.device)
        
        # --- Save Model, Configs and Loss ---
        
        if args.save:
            save_model(model, str(args.epochs), f'Transformer_N{args.N}')
            save_configs(en_vocab, pt_vocab, args, f'Transformer_N{args.N}')
            plot_loss(args.epochs, loss, f'Transformer_N{args.N}')
    
    elif args.mode == 'inference':
        
        configs, eng, pt, en_vocab, pt_vocab = load_configs(args.model_config_path)

        model = Transformer(n=args.N,
                            dm=args.dm,
                            dff=args.dff,
                            heads=args.heads,
                            source_vocab=en_vocab,
                            target_vocab=pt_vocab,
                            device=args.device
        ).to(args.device)
        
        #TODO autoregressive generator 
        
