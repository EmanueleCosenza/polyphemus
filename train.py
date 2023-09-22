import torch.optim as optim
import matplotlib.pyplot as plt
import uuid
import copy
import time
from statistics import mean
from collections import defaultdict
import math
import torch
from torch import nn, Tensor
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F


def generate_square_subsequent_mask(sz):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def append_dict(dest_d, source_d):
        
    for k, v in source_d.items():
        dest_d[k].append(v)


class VAETrainer():
    
    def __init__(self, model_dir, checkpoint=False, model=None, optimizer=None,
                 init_lr=1e-4, lr_scheduler=None, device=torch.device("cuda"), 
                 print_every=1, save_every=1, eval_every=100, iters_to_accumulate=1,
                 **kwargs):
        
        self.__dict__.update(kwargs)
        
        self.model_dir = model_dir
        self.device = device
        self.print_every = print_every
        self.save_every = save_every
        self.eval_every = eval_every
        self.iters_to_accumulate = iters_to_accumulate
        
        # Criteria with ignored padding
        self.bce_unreduced = nn.BCEWithLogitsLoss(reduction='none')
        self.ce_p = nn.CrossEntropyLoss(ignore_index=130)
        self.ce_d = nn.CrossEntropyLoss(ignore_index=98)
        
        # Training stats
        self.tr_losses = defaultdict(list)
        self.tr_accuracies = defaultdict(list)
        self.val_losses = defaultdict(list)
        self.val_accuracies = defaultdict(list)
        self.lrs = []
        self.betas = []
        self.times = []
        
        self.model = model
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.lr_scheduler = lr_scheduler
        
        self.tot_batches = 0
        self.beta = 0
        self.min_val_loss = np.inf
        
        if checkpoint:
            self.load_checkpoint()
        
    
    def train(self, trainloader, validloader=None, epochs=1,
              early_exit=None):
        
        self.model.train()
        
        print("Starting training.\n")
        
        if not self.times:
            start = time.time()
            self.times.append(start)
        
        progress_bar = tqdm(range(len(trainloader)))
        scaler = torch.cuda.amp.GradScaler()
                
        # Zero out the gradients
        self.optimizer.zero_grad()
        
        for epoch in range(epochs):
            
            self.cur_epoch = epoch
            
            for batch_idx, inputs in enumerate(trainloader):
                
                self.cur_batch_idx = batch_idx
                
                # Get the inputs
                x_graph = inputs.to(self.device)
                x_seq, x_acts, src_mask = x_graph.x_seq, x_graph.x_acts, x_graph.src_mask
                tgt_mask = generate_square_subsequent_mask(x_seq.size(-2)-1).to(self.device)
                
                inputs = (x_seq, x_acts, x_graph)

                with torch.cuda.amp.autocast():
                    # Forward pass, get the reconstructions
                    outputs, mu, log_var = self.model(x_seq, x_acts, x_graph, src_mask, tgt_mask)

                    # Compute the backprop loss and other required losses
                    tot_loss, losses = self._compute_losses(inputs, outputs, mu,
                                                             log_var)
                    tot_loss = tot_loss / self.iters_to_accumulate
                
                # Free GPU
                del x_seq
                del x_acts
                del src_mask
                del tgt_mask
                
                # Backprop
                scaler.scale(tot_loss).backward()
                #tot_loss.backward()
                #self.optimizer.step()
                    
                if (self.tot_batches + 1) % self.iters_to_accumulate == 0:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    if self.beta_update:    
                        self._update_beta()
                
                # Compute accuracies
                accs = self._compute_accuracies(inputs, outputs, x_graph.is_drum)
                
                # Update the stats
                append_dict(self.tr_losses, losses)
                last_lr = (self.lr_scheduler.lr 
                               if self.lr_scheduler is not None else self.init_lr)
                self.lrs.append(last_lr)
                self.betas.append(self.beta)
                append_dict(self.tr_accuracies, accs)
                now = time.time()
                self.times.append(now)
                
                # Print stats
                if (self.tot_batches + 1) % self.print_every == 0:
                    print("Training on batch {}/{} of epoch {}/{} complete."
                          .format(batch_idx+1, len(trainloader), epoch+1, epochs))
                    self._print_stats()
                    print("\n----------------------------------------\n")
                
                # Eval on VL set every `n` gradient updates
                if validloader is not None and (self.tot_batches + 1) % self.eval_every == 0:
                    
                    # Evaluate on val set
                    print("\nEvaluating on validation set...\n")
                    val_losses, val_accuracies = self.evaluate(validloader)
                    
                    # Update stats
                    append_dict(self.val_losses, val_losses)
                    append_dict(self.val_accuracies, val_accuracies)
                    
                    print("Val losses:")
                    print(val_losses)
                    print("Val accuracies:")
                    print(val_accuracies)
                    
                    # Save model if val loss (tot) reached a new minimum
                    tot_loss = val_losses['tot']
                    if tot_loss < self.min_val_loss:
                        print("\nValidation loss improved.")
                        print("Saving new best model to disk...\n")
                        self._save_model('best_model')
                        self.min_val_loss = tot_loss
                    
                    self.model.train()
                
                progress_bar.update(1)     
                    
                # When appropriate, save model and stats on disk
                if self.save_every > 0 and (self.tot_batches + 1) % self.save_every == 0:
                    print("\nSaving model to disk...\n")
                    self._save_model('checkpoint')
                
                # Stop prematurely if early_exit is set and reached
                if early_exit is not None and (self.tot_batches + 1) > early_exit:
                    break
                
                self.tot_batches += 1
            

        end = time.time()
        # Todo: self.__print_time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Training completed in (h:m:s): {:0>2}:{:0>2}:{:05.2f}"
                  .format(int(hours),int(minutes),seconds))
        
        print("Saving model to disk...")
        self._save_model('checkpoint')
        
        print("Model saved.")
        
    
    def _update_beta(self):
        
        # Number of gradient updates
        i = self.tot_batches
        
        if i < self.anneal_start or i >= self.anneal_end:
            return
        
        n_steps = self.beta_max // self.step_size
        inc_every = (self.anneal_end - self.anneal_start) // n_steps
        
        curr_step = (i - self.anneal_start) // inc_every
        self.beta = self.step_size * (curr_step + 1)
        
    
    def evaluate(self, loader):
        
        losses = defaultdict(list)
        accs = defaultdict(list)
        
        self.model.eval()
        progress_bar = tqdm(range(len(loader)))
        
        with torch.no_grad():
            for batch_idx, inputs in enumerate(loader):

                # Get the inputs and move them to device
                x_graph = inputs.to(self.device)
                x_seq, x_acts, src_mask = x_graph.x_seq, x_graph.x_acts, x_graph.src_mask
                tgt_mask = generate_square_subsequent_mask(x_seq.size(-2)-1).to(self.device)
                inputs = (x_seq, x_acts, x_graph)

                with torch.cuda.amp.autocast():
                    # Forward pass, get the reconstructions
                    outputs, mu, log_var = self.model(x_seq, x_acts, x_graph, src_mask, tgt_mask)

                    # Compute losses and accuracies wrt batch
                    _, losses_b = self._compute_losses(inputs, outputs, mu,
                                                         log_var)
                    
                accs_b = self._compute_accuracies(inputs, outputs, x_graph.is_drum)
                
                # Save losses and accuracies
                append_dict(losses, losses_b)
                append_dict(accs, accs_b)
                
                progress_bar.update(1)
        
        
        # Compute avg losses and accuracies
        avg_losses = {}
        for k, l in losses.items():
            avg_losses[k] = mean(l)
            
        avg_accs = {}
        for k, l in accs.items():
            avg_accs[k] = mean(l)
            
        return avg_losses, avg_accs
                
        
    
    def _compute_losses(self, inputs, outputs, mu, log_var):
        
        x_seq, x_acts, _ = inputs
        seq_rec, acts_rec = outputs
        
        # Shift outputs for transformer decoder loss and filter silences
        x_seq = x_seq[..., 1:, :]
                
        # Compute the losses
        acts_loss = self.bce_unreduced(acts_rec.view(-1), x_acts.view(-1).float())
        acts_loss = torch.mean(acts_loss)
        
        pitches_loss = self.ce_p(seq_rec.reshape(-1, seq_rec.size(-1))[:, :131],
                          x_seq.reshape(-1, x_seq.size(-1))[:, :131].argmax(dim=1))
        dur_loss = self.ce_d(seq_rec.reshape(-1, seq_rec.size(-1))[:, 131:],
                          x_seq.reshape(-1, x_seq.size(-1))[:, 131:].argmax(dim=1))
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        kld_loss = torch.mean(kld_loss)
        rec_loss = pitches_loss + dur_loss + acts_loss
        tot_loss = rec_loss + self.beta*kld_loss
        
        losses = {
            'tot': tot_loss.item(),
            'pitches': pitches_loss.item(),
            'dur': dur_loss.item(),
            'acts': acts_loss.item(),
            'rec': rec_loss.item(),
            'kld': kld_loss.item(),
            'beta*kld': self.beta*kld_loss.item()
        }
        
        return tot_loss, losses
            
            
    def _compute_accuracies(self, inputs, outputs, is_drum):
        
        x_seq, x_acts, _ = inputs
        seq_rec, acts_rec = outputs
        
        # Shift outputs and filter silences
        x_seq = x_seq[..., 1:, :]
        
        notes_acc = self._note_accuracy(seq_rec, x_seq)
        pitches_acc = self._pitches_accuracy(seq_rec, x_seq)
        pitches_acc_drums = self._pitches_accuracy(seq_rec, x_seq, 
                                                   is_drum, drums=True)
        pitches_acc_non_drums = self._pitches_accuracy(seq_rec, x_seq,
                                                       is_drum, drums=False)
        dur_acc = self._dur_accuracy(seq_rec, x_seq)
        acts_acc = self._acts_accuracy(acts_rec, x_acts)
        acts_precision = self._acts_precision(acts_rec, x_acts)
        acts_recall = self._acts_recall(acts_rec, x_acts)
        acts_f1 = (2 * acts_recall * acts_precision / 
                       (acts_recall + acts_precision))
        
        accs = {
            'notes': notes_acc.item(),
            'pitches': pitches_acc.item(),
            'pitches_drums': pitches_acc_drums.item(),
            'pitches_non_drums': pitches_acc_non_drums.item(),
            'dur': dur_acc.item(),
            'acts_acc': acts_acc.item(),
            'acts_precision': acts_precision.item(),
            'acts_recall': acts_recall.item(),
            'acts_f1': acts_f1.item()
        }
        
        return accs
    
    
    def _note_accuracy(self, seq_rec, x_seq):
        
        pitches_rec = F.softmax(seq_rec[..., :131], dim=-1)
        pitches_rec = torch.argmax(pitches_rec, dim=-1)
        pitches_true = torch.argmax(x_seq[..., :131], dim=-1)
        
        mask_p = (pitches_true != 130)
        
        preds_pitches = (pitches_rec == pitches_true)
        preds_pitches = torch.logical_and(preds_pitches, mask_p)
        
        
        dur_rec = F.softmax(seq_rec[..., 131:], dim=-1)
        dur_rec = torch.argmax(dur_rec, dim=-1)
        dur_true = torch.argmax(x_seq[..., 131:], dim=-1)
        
        mask_d = (dur_true != 98)
        
        preds_dur = (dur_rec == dur_true)
        preds_dur = torch.logical_and(preds_dur, mask_d)
        
        return torch.sum(torch.logical_and(preds_pitches, 
                                           preds_dur)) / torch.sum(mask_p)
    
    
    def _acts_precision(self, acts_rec, x_acts):
        
        acts_rec = torch.sigmoid(acts_rec)
        acts_rec[acts_rec < 0.5] = 0
        acts_rec[acts_rec >= 0.5] = 1
        
        tp = torch.sum(x_acts[acts_rec == 1])
        
        return tp / torch.sum(acts_rec)
    
    
    def _acts_recall(self, acts_rec, x_acts):
        
        acts_rec = torch.sigmoid(acts_rec)
        acts_rec[acts_rec < 0.5] = 0
        acts_rec[acts_rec >= 0.5] = 1
        
        tp = torch.sum(x_acts[acts_rec == 1])
        
        return tp / torch.sum(x_acts)
    
    
    def _acts_accuracy(self, acts_rec, x_acts):
        
        acts_rec = torch.sigmoid(acts_rec)
        acts_rec[acts_rec < 0.5] = 0
        acts_rec[acts_rec >= 0.5] = 1
        
        return torch.sum(acts_rec == x_acts) / x_acts.numel()
    
    
    def _pitches_accuracy(self, seq_rec, x_seq, is_drum=None, drums=None):
        
        if drums is not None:
            if drums:
                seq_rec = seq_rec[is_drum]
                x_seq = x_seq[is_drum]
            else:
                seq_rec = seq_rec[torch.logical_not(is_drum)]
                x_seq = x_seq[torch.logical_not(is_drum)]
        
        pitches_rec = F.softmax(seq_rec[..., :131], dim=-1)
        pitches_rec = torch.argmax(pitches_rec, dim=-1)
        pitches_true = torch.argmax(x_seq[..., :131], dim=-1)
        
        mask = (pitches_true != 130)
        
        preds_pitches = (pitches_rec == pitches_true)
        preds_pitches = torch.logical_and(preds_pitches, mask)
        
        return torch.sum(preds_pitches) / torch.sum(mask)
    
    
    def _dur_accuracy(self, seq_rec, x_seq):
        
        dur_rec = F.softmax(seq_rec[..., 131:], dim=-1)
        dur_rec = torch.argmax(dur_rec, dim=-1)
        dur_true = torch.argmax(x_seq[..., 131:], dim=-1)
        
        mask = (dur_true != 98)
        
        preds_dur = (dur_rec == dur_true)
        preds_dur = torch.logical_and(preds_dur, mask)
        
        return torch.sum(preds_dur) / torch.sum(mask)
    
    
    def _save_model(self, filename):
        path = os.path.join(self.model_dir, filename)
        torch.save({
            'epoch': self.cur_epoch,
            'batch': self.cur_batch_idx,
            'tot_batches': self.tot_batches,
            'betas': self.betas,
            'min_val_loss': self.min_val_loss,
            'print_every': self.print_every,
            'save_every': self.save_every,
            'eval_every': self.eval_every,
            'lrs': self.lrs,
            'tr_losses': self.tr_losses,
            'tr_accuracies': self.tr_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        
    
    def load(self):
        
        checkpoint = torch.load(os.path.join(self.model_dir, 'checkpoint'))
        
        self.cur_epoch = checkpoint['epoch']
        self.cur_batch_idx = checkpoint['batch']
        self.save_every = checkpoint['save_every']
        self.eval_every = checkpoint['eval_every']
        self.lrs = checkpoint['lrs']
        self.tr_losses = checkpoint['tr_losses']
        self.tr_accuracies = checkpoint['tr_accuracies']
        self.val_losses = checkpoint['val_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        self.times = checkpoint['times']
        self.min_val_loss = checkpoint['min_val_loss']
        self.beta = checkpoint['beta']
        self.tot_batches = checkpoint['tot_batches']
        
        
    def _print_stats(self):
        
        hours, rem = divmod(self.times[-1]-self.times[0], 3600)
        minutes, seconds = divmod(rem, 60)
        print("Elapsed time from start (h:m:s): {:0>2}:{:0>2}:{:05.2f}"
                  .format(int(hours), int(minutes), seconds))
        
        avg_lr = mean(self.lrs[-self.print_every:])
        
        # Take mean of the last non-printed batches for each stat
        
        avg_losses = {}
        for k, l in self.tr_losses.items():
            avg_losses[k] = mean(l[-self.print_every:])
        
        avg_accs = {}
        for k, l in self.tr_accuracies.items():
            avg_accs[k] = mean(l[-self.print_every:])
        
        print("Losses:")
        print(avg_losses)
        print("Accuracies:")
        print(avg_accs)
        
