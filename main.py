import os
import shutil
import yaml
from types import SimpleNamespace
import torch
import numpy as np
import random

from torch import optim
from torch.utils.tensorboard import SummaryWriter

from data import EgoviewsDataset, DataLoader
from model import PoseCorrectionNet
from loss import PoseLoss


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
    shutil.copy(config.self_path, config.save_config_path)


def load_config(path):
    with open(path, 'r') as file:
        config = SimpleNamespace(**yaml.load(file, Loader=yaml.FullLoader))

        config.experiment_dir = os.path.join(config.log_dir, config.experiment_name)
        config.tb_dir = os.path.join(config.experiment_dir, 'tb')
        config.save_model_path = os.path.join(config.experiment_dir, 'model.pt')
        config.save_config_path = os.path.join(config.experiment_dir, 'config.yaml')
        config.self_path = path

    return config


def train(config, model, optimizer, train_loader, loss_f, device, epoch, writer):
    model.train()

    total_weighted_loss = 0
    total_loc_loss = 0
    total_orient_loss = 0
    for data, target in train_loader:
        observed_egoview, expected_egoview = data

        observed_egoview = observed_egoview.to(device)
        expected_egoview = expected_egoview.to(device)
        target = target.to(device)

        output = model(observed_egoview, expected_egoview)
        weighted_loss, loc_loss, orient_loss = loss_f(output, target)

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        total_weighted_loss += weighted_loss.item()
        total_loc_loss += loc_loss.item()
        total_orient_loss += orient_loss.item()

    num_batches = len(train_loader)
    avg_weighted_loss = total_weighted_loss / num_batches
    avg_loc_loss = total_loc_loss / num_batches
    avg_orient_loss = total_orient_loss / num_batches

    print(
        'Train Epoch: {} \t'
        'weighted_loss: {:.6f} \t'
        'location_loss: {:.6f} \t'
        'orientation_loss: {:.6f}'.format(epoch, avg_weighted_loss, avg_loc_loss, avg_orient_loss)
    )

    writer.add_scalar(f'{config.experiment_name}/weighted_loss', avg_weighted_loss, epoch)
    writer.add_scalar(f'{config.experiment_name}/location_loss', avg_loc_loss, epoch)
    writer.add_scalar(f'{config.experiment_name}/orientation_loss', avg_orient_loss, epoch)


def val(config, model, val_loader, loss_f, device, epoch, writer):
    model.eval()

    total_weighted_loss = 0
    total_loc_loss = 0
    total_orient_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            observed_egoview, expected_egoview = data

            observed_egoview = observed_egoview.to(device)
            expected_egoview = expected_egoview.to(device)
            target = target.to(device)

            output = model(observed_egoview, expected_egoview)
            weighted_loss, loc_loss, orient_loss = loss_f(output, target)

            total_weighted_loss += weighted_loss.item()
            total_loc_loss += loc_loss.item()
            total_orient_loss += orient_loss.item()

    num_batches = len(val_loader)
    avg_weighted_loss = total_weighted_loss / num_batches
    avg_loc_loss = total_loc_loss / num_batches
    avg_orient_loss = total_orient_loss / num_batches

    print(
        'Val set: '
        'weighted_loss: {:.6f} \t'
        'location_loss: {:.6f} \t'
        'orientation_loss: {:.6f}'.format(avg_weighted_loss, avg_loc_loss, avg_orient_loss)
    )

    writer.add_scalar(f'{config.experiment_name}/weighted_loss', avg_weighted_loss, epoch)
    writer.add_scalar(f'{config.experiment_name}/location_loss', avg_loc_loss, epoch)
    writer.add_scalar(f'{config.experiment_name}/orientation_loss', avg_orient_loss, epoch)


def main():
    config_path = './config.yaml'
    config = load_config(config_path)

    init_experiment(config)
    set_random_seed(config.seed)

    train_dataset = EgoviewsDataset(config.train_data_path)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    val_dataset = EgoviewsDataset(config.val_data_path)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.val_batch_size,
        num_workers=config.num_workers
    )

    device = torch.device(config.device)

    model = PoseCorrectionNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = None
    loss_f = PoseLoss(alpha=1., beta=10.)

    train_writer = SummaryWriter(log_dir=os.path.join(config.tb_dir, 'train'))
    val_writer = SummaryWriter(log_dir=os.path.join(config.tb_dir, 'val'))

    for epoch in range(1, config.epochs + 1):
        train(config, model, optimizer, train_loader, loss_f, device, epoch, train_writer)
        val(config, model, val_loader, loss_f, device, epoch, val_writer)

        if scheduler:
            scheduler.step()

    train_writer.close()
    val_writer.close()

    if config.save_model:
        torch.save(model.state_dict(), config.save_model_path)


if __name__ == '__main__':
    main()

