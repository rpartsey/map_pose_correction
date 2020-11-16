import os
import random
import shutil
from types import SimpleNamespace

import numpy as np
import torch
import yaml


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=1e-7, path='checkpoint.pt', trace_func=print, save=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.save = save

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif self.best_score - score < self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        if self.save:
            self.trace_func('Saving model ...')
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class RecursiveNamespace(SimpleNamespace):
    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)

        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if val is None:
                setattr(self, key, RecursiveNamespace())
            elif type(val) == dict:
                setattr(self, key, RecursiveNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))


def write_metrics(epoch, metrics, writer):
    avg_weighted_loss = metrics['avg_weighted_loss']
    avg_loc_loss = metrics['avg_loc_loss']
    avg_orient_loss = metrics['avg_orient_loss']

    writer.add_scalar('metrics/weighted_loss', avg_weighted_loss, epoch)
    writer.add_scalar('metrics/location_loss', avg_loc_loss, epoch)
    writer.add_scalar('metrics/orientation_loss', avg_orient_loss, epoch)


def print_metrics(phase, metrics):
    avg_weighted_loss = metrics['avg_weighted_loss']
    avg_loc_loss = metrics['avg_loc_loss']
    avg_orient_loss = metrics['avg_orient_loss']

    print(
        '{:6}'
        'weighted_loss: {:.6f} \t'
        'location_loss: {:.6f} \t'
        'orientation_loss: {:.6f}'.format(phase, avg_weighted_loss, avg_loc_loss, avg_orient_loss)
    )


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_experiment(config):
    if os.path.exists(config.experiment_dir):
        def ask():
            return input(f'Experiment "{config.experiment_name}" already exists. Delete (y/n)?')

        answer = ask()
        while answer not in ('y', 'n'):
            answer = ask()

        delete = answer == 'y'
        if not delete:
            exit(1)

        shutil.rmtree(config.experiment_dir)

    os.makedirs(config.experiment_dir)
    shutil.copy(config.self_path, config.config_save_path)


def load_config(path):
    with open(path, 'r') as file:
        config = RecursiveNamespace(**yaml.load(file, Loader=yaml.FullLoader))

        config.experiment_dir = os.path.join(config.log_dir, config.experiment_name)
        config.tb_dir = os.path.join(config.experiment_dir, 'tb')
        config.model.save_path = os.path.join(config.experiment_dir, 'checkpoint.pt')
        config.config_save_path = os.path.join(config.experiment_dir, 'config.yaml')
        config.self_path = path

    return config
