import time
import os
from statistics import mean
from collections import defaultdict

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm
import pprint
import math

import constants
from constants import PitchToken, DurationToken
from utils import append_dict, print_divider


class StepBetaScheduler():
    def __init__(self, anneal_start, beta_max, step_size, anneal_end):
        self.anneal_start = anneal_start
        self.beta_max = beta_max
        self.step_size = step_size
        self.anneal_end = anneal_end

        self.update_steps = 0
        self.beta = 0
        n_steps = self.beta_max // self.step_size
        self.inc_every = (self.anneal_end-self.anneal_start) // n_steps

    def step(self):
        self.update_steps += 1

        if (self.update_steps >= self.anneal_start or
                self.update_steps < self.anneal_end):
            # If we are annealing, update beta according to current step
            curr_step = (self.update_steps-self.anneal_start) // self.inc_every
            self.beta = self.step_size * (curr_step+1)
            
        return self.beta


class ExpDecayLRScheduler():
    def __init__(self, optimizer, peak_lr, warmup_steps, final_lr_scale,
                 decay_steps):

        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

        # Find the decay factor needed to reach the specified
        # learning rate scale after decay_steps steps
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

        self.update_steps = 0

    def set_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def step(self):
        self.update_steps += 1

        if self.update_steps <= self.warmup_steps:
            self.lr = self.peak_lr
        else:
            # Decay lr exponentially
            steps_after_warmup = self.update_steps - self. warmup_steps
            self.lr = \
                self.peak_lr * math.exp(-self.decay_factor*steps_after_warmup)

        self.set_lr(self.optimizer, self.lr)

        return self.lr


class PolyphemusTrainer():

    def __init__(self, model_dir, model, optimizer, init_lr=1e-4,
                 lr_scheduler=None, beta_scheduler=None, device=None, 
                 print_every=1, save_every=1, eval_every=100, 
                 iters_to_accumulate=1, **kwargs):
        self.__dict__.update(kwargs)

        self.model_dir = model_dir
        self.model = model
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.lr_scheduler = lr_scheduler
        self.beta_scheduler = beta_scheduler
        self.device = device if device is not None else torch.device("cpu")
        self.cuda = True if self.device.type == 'cuda' else False
        self.print_every = print_every
        self.save_every = save_every
        self.eval_every = eval_every
        self.iters_to_accumulate = iters_to_accumulate

        # Losses (ignoring PAD tokens)
        self.bce_unreduced = nn.BCEWithLogitsLoss(reduction='none')
        self.ce_p = nn.CrossEntropyLoss(ignore_index=PitchToken.PAD.value)
        self.ce_d = nn.CrossEntropyLoss(ignore_index=DurationToken.PAD.value)

        # Training stats
        self.tr_losses = defaultdict(list)
        self.tr_accuracies = defaultdict(list)
        self.val_losses = defaultdict(list)
        self.val_accuracies = defaultdict(list)
        self.lrs = []
        self.betas = []
        self.times = []

    def train(self, trainloader, validloader=None, epochs=100, early_exit=None):

        self.tot_batches = 0
        self.beta = 0
        self.min_val_loss = np.inf

        start = time.time()
        self.times.append(start)

        self.model.train()
        scaler = torch.cuda.amp.GradScaler() if self.cuda else None
        self.optimizer.zero_grad()
        progress_bar = tqdm(range(len(trainloader)))

        for epoch in range(epochs):
            self.cur_epoch = epoch
            for batch_idx, graph in enumerate(trainloader):
                self.cur_batch_idx = batch_idx

                # Move batch of graphs to device. Note: a single graph here
                # represents a bar in the original sequence.
                graph = graph.to(self.device)
                s_tensor, c_tensor = graph.s_tensor, graph.c_tensor

                with torch.cuda.amp.autocast(enabled=self.cuda):
                    # Forward pass to obtain mu, log(sigma^2), computed by the
                    # encoder, and structure and content logits, computed by the
                    # decoder
                    (s_logits, c_logits), mu, log_var = self.model(graph)

                    # Compute losses
                    tot_loss, losses = self._losses(
                        s_tensor, s_logits,
                        c_tensor, c_logits,
                        mu, log_var
                    )
                    tot_loss = tot_loss / self.iters_to_accumulate

                # Backpropagation
                if self.cuda:
                    scaler.scale(tot_loss).backward()
                else:
                    tot_loss.backward()

                # Update weights with accumulated gradients
                if (self.tot_batches + 1) % self.iters_to_accumulate == 0:

                    if self.cuda:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad()

                    # Update lr and beta
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    if self.beta_scheduler is not None:
                        self.beta_scheduler.step()

                # Compute accuracies
                accs = self._accuracies(
                    s_tensor, s_logits,
                    c_tensor, c_logits,
                    graph.is_drum
                )

                # Update the stats
                append_dict(self.tr_losses, losses)
                append_dict(self.tr_accuracies, accs)
                last_lr = (self.lr_scheduler.lr
                           if self.lr_scheduler is not None else self.init_lr)
                self.lrs.append(last_lr)
                self.betas.append(self.beta)
                now = time.time()
                self.times.append(now)

                # Print stats
                if (self.tot_batches + 1) % self.print_every == 0:
                    print("Training on batch {}/{} of epoch {}/{} complete."
                          .format(batch_idx+1,
                                  len(trainloader),
                                  epoch+1,
                                  epochs))
                    self._print_stats()
                    print_divider()

                # Eval on VL every `eval_every` gradient updates
                if (validloader is not None and
                        (self.tot_batches + 1) % self.eval_every == 0):

                    # Evaluate on VL
                    print("\nEvaluating on validation set...\n")
                    val_losses, val_accuracies = self.evaluate(validloader)

                    # Update stats
                    append_dict(self.val_losses, val_losses)
                    append_dict(self.val_accuracies, val_accuracies)

                    print("Val losses:")
                    print(val_losses)
                    print("Val accuracies:")
                    print(val_accuracies)

                    # Save model if VL loss (tot) reached a new minimum
                    tot_loss = val_losses['tot']
                    if tot_loss < self.min_val_loss:
                        print("\nValidation loss improved.")
                        print("Saving new best model to disk...\n")
                        self._save_model('best_model')
                        self.min_val_loss = tot_loss

                    self.model.train()

                progress_bar.update(1)

                # Save model and stats on disk
                if (self.save_every > 0 and
                        (self.tot_batches + 1) % self.save_every == 0):
                    self._save_model('checkpoint')

                # Stop prematurely if early_exit is set and reached
                if (early_exit is not None and
                        (self.tot_batches + 1) > early_exit):
                    break

                self.tot_batches += 1

        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Training completed in (h:m:s): {:0>2}:{:0>2}:{:05.2f}"
              .format(int(hours), int(minutes), seconds))

        self._save_model('checkpoint')

        print("Model saved.")

    def evaluate(self, loader):

        losses = defaultdict(list)
        accs = defaultdict(list)

        self.model.eval()
        progress_bar = tqdm(range(len(loader)))

        with torch.no_grad():
            for _, graph in enumerate(loader):

                # Get the inputs and move them to device
                graph = graph.to(self.device)
                s_tensor, c_tensor = graph.s_tensor, graph.c_tensor

                with torch.cuda.amp.autocast():
                    # Forward pass, get the reconstructions
                    (s_logits, c_logits), mu, log_var = self.model(graph)

                    _, losses_b = self._losses(
                        s_tensor, s_logits,
                        c_tensor, c_logits,
                        mu, log_var
                    )

                accs_b = self._accuracies(
                    s_tensor, s_logits,
                    c_tensor, c_logits,
                    graph.is_drum
                )

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

    def _losses(self, s_tensor, s_logits, c_tensor, c_logits, mu, log_var):

        # Do not consider SOS token
        c_tensor = c_tensor[..., 1:, :]
        c_logits = c_logits.reshape(-1, c_logits.size(-1))
        c_tensor = c_tensor.reshape(-1, c_tensor.size(-1))

        # Reshape logits to match s_tensor dimensions:
        # n_graphs (in batch) x n_tracks x n_timesteps
        s_logits = s_tensor.reshape(-1, *s_logits.shape[2:])

        # Binary structure tensor loss (binary cross entropy)
        s_loss = self.bce_unreduced(
            s_logits.view(-1), s_tensor.view(-1).float())
        s_loss = torch.mean(s_loss)

        # Content tensor loss (pitches). argmax is used to obtain token numbers
        # from onehot rep
        pitch_logits = c_logits[:, :constants.N_PITCH_TOKENS]
        pitch_true = c_tensor[:, :constants.N_PITCH_TOKENS].argmax(dim=1)
        pitch_loss = self.ce_p(pitch_logits, pitch_true)

        # Content tensor loss (durations)
        dur_logits = c_logits[:, constants.N_PITCH_TOKENS:]
        dur_true = c_tensor[:, constants.N_PITCH_TOKENS:].argmax(dim=1)
        dur_loss = self.ce_d(dur_logits, dur_true)

        # Kullback-Leibler divergence loss
        # Derivation in Kingma, Diederik P., and Max Welling. "Auto-encoding
        # variational bayes." (2013), Appendix B.
        # (https://arxiv.org/pdf/1312.6114.pdf)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),
                                    dim=1)
        kld_loss = torch.mean(kld_loss)

        # Reconstruction loss and total loss
        rec_loss = pitch_loss + dur_loss + s_loss
        tot_loss = rec_loss + self.beta*kld_loss

        losses = {
            'tot': tot_loss.item(),
            'pitch': pitch_loss.item(),
            'dur': dur_loss.item(),
            'structure': s_loss.item(),
            'reconstruction': rec_loss.item(),
            'kld': kld_loss.item(),
            'beta*kld': self.beta*kld_loss.item()
        }

        return tot_loss, losses

    def _accuracies(self, s_tensor, s_logits, c_tensor, c_logits, is_drum):

        # Do not consider SOS token
        c_tensor = c_tensor[..., 1:, :]

        # Reshape logits to match s_tensor dimensions:
        # n_graphs (in batch) x n_tracks x n_timesteps
        s_logits = s_tensor.reshape(-1, *s_logits.shape[2:])

        # Note accuracy considers both pitches and durations
        note_acc = self._note_accuracy(c_logits, c_tensor)

        pitch_acc = self._pitch_accuracy(c_logits, c_tensor)

        # Compute pitch accuracies for drums and non drums separately
        pitch_acc_drums = self._pitch_accuracy(
            c_logits, c_tensor, drums=True, is_drum=is_drum
        )
        pitch_acc_non_drums = self._pitch_accuracy(
            c_logits, c_tensor, drums=False, is_drum=is_drum
        )

        dur_acc = self._duration_accuracy(c_logits, c_tensor)

        s_acc = self._structure_accuracy(s_logits, s_tensor)
        s_precision = self._structure_precision(s_logits, s_tensor)
        s_recall = self._structure_recall(s_logits, s_tensor)
        s_f1 = (2*s_recall*s_precision / (s_recall+s_precision))

        accs = {
            'note': note_acc.item(),
            'pitch': pitch_acc.item(),
            'pitch_drums': pitch_acc_drums.item(),
            'pitch_non_drums': pitch_acc_non_drums.item(),
            'dur': dur_acc.item(),
            's_acc': s_acc.item(),
            's_precision': s_precision.item(),
            's_recall': s_recall.item(),
            's_f1': s_f1.item()
        }

        return accs

    def _pitch_accuracy(self, c_logits, c_tensor, drums=None, is_drum=None):

        # When drums is None, just compute the global pitch accuracy without
        # distinguishing between drum and non drum pitches
        if drums is not None:
            if drums:
                c_logits = c_logits[is_drum]
                c_tensor = c_tensor[is_drum]
            else:
                c_logits = c_logits[torch.logical_not(is_drum)]
                c_tensor = c_tensor[torch.logical_not(is_drum)]

        # Apply softmax to obtain pitch reconstructions
        pitch_rec = c_logits[..., :constants.N_PITCH_TOKENS]
        pitch_rec = F.softmax(pitch_rec, dim=-1)
        pitch_rec = torch.argmax(pitch_rec, dim=-1)

        pitch_true = c_tensor[..., :constants.N_PITCH_TOKENS]
        pitch_true = torch.argmax(pitch_true, dim=-1)

        # Do not consider PAD tokens when computing accuracies
        not_pad = (pitch_true != PitchToken.PAD.value)

        correct = (pitch_rec == pitch_true)
        correct = torch.logical_and(correct, not_pad)

        return torch.sum(correct) / torch.sum(not_pad)

    def _duration_accuracy(self, c_logits, c_tensor):

        # Apply softmax to obtain reconstructed durations
        dur_rec = c_logits[..., constants.N_PITCH_TOKENS:]
        dur_rec = F.softmax(dur_rec, dim=-1)
        dur_rec = torch.argmax(dur_rec, dim=-1)

        dur_true = c_tensor[..., constants.N_PITCH_TOKENS:]
        dur_true = torch.argmax(dur_true, dim=-1)

        # Do not consider PAD tokens when computing accuracies
        not_pad = (dur_true != DurationToken.PAD.value)

        correct = (dur_rec == dur_true)
        correct = torch.logical_and(correct, not_pad)

        return torch.sum(correct) / torch.sum(not_pad)

    def _note_accuracy(self, c_logits, c_tensor):

        # Apply softmax to obtain pitch reconstructions
        pitch_rec = c_logits[..., :constants.N_PITCH_TOKENS]
        pitch_rec = F.softmax(pitch_rec, dim=-1)
        pitch_rec = torch.argmax(pitch_rec, dim=-1)

        pitch_true = c_tensor[..., :constants.N_PITCH_TOKENS]
        pitch_true = torch.argmax(pitch_true, dim=-1)

        not_pad_p = (pitch_true != PitchToken.PAD.value)

        correct_p = (pitch_rec == pitch_true)
        correct_p = torch.logical_and(correct_p, not_pad_p)

        dur_rec = c_logits[..., constants.N_PITCH_TOKENS:]
        dur_rec = F.softmax(dur_rec, dim=-1)
        dur_rec = torch.argmax(dur_rec, dim=-1)

        dur_true = c_tensor[..., constants.N_PITCH_TOKENS:]
        dur_true = torch.argmax(dur_true, dim=-1)

        not_pad_d = (dur_true != DurationToken.PAD.value)

        correct_d = (dur_rec == dur_true)
        correct_d = torch.logical_and(correct_d, not_pad_d)

        note_accuracy = torch.sum(
            torch.logical_and(correct_p, correct_d)) / torch.sum(not_pad_p)

        return note_accuracy

    def _structure_accuracy(self, s_logits, s_tensor):

        s_logits = torch.sigmoid(s_logits)
        s_logits[s_logits < 0.5] = 0
        s_logits[s_logits >= 0.5] = 1

        return torch.sum(s_logits == s_tensor) / s_tensor.numel()

    def _structure_precision(self, s_logits, s_tensor):

        s_logits = torch.sigmoid(s_logits)
        s_logits[s_logits < 0.5] = 0
        s_logits[s_logits >= 0.5] = 1

        tp = torch.sum(s_tensor[s_logits == 1])

        return tp / torch.sum(s_logits)

    def _structure_recall(self, s_logits, s_tensor):

        s_logits = torch.sigmoid(s_logits)
        s_logits[s_logits < 0.5] = 0
        s_logits[s_logits >= 0.5] = 1

        tp = torch.sum(s_tensor[s_logits == 1])

        return tp / torch.sum(s_tensor)

    def _save_model(self, filename):

        path = os.path.join(self.model_dir, filename)
        print("Saving model to disk...")

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

        print("The model has been successfully saved.")

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

        # Take mean of the last non-printed batches for each loss and accuracy
        avg_losses = {}
        for k, l in self.tr_losses.items():
            v = mean(l[-self.print_every:])
            avg_losses[k] = round(v, 2)

        avg_accs = {}
        for k, l in self.tr_accuracies.items():
            v = mean(l[-self.print_every:])
            avg_accs[k] = round(v, 2)

        print("Losses:")
        pprint.pprint(avg_losses, indent=2)

        print("Accuracies:")
        pprint.pprint(avg_accs, indent=2)
