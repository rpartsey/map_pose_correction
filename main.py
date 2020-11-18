import os

import torch
from torch.utils.tensorboard import SummaryWriter

import models
import data
import optims
import losses
from utils import (
    EarlyStopping,
    write_metrics,
    print_metrics,
    init_experiment,
    set_random_seed,
    load_config
)


def train(model, optimizer, train_loader, loss_f, device):
    model.train()

    total_weighted_loss = 0
    total_loc_loss = 0
    total_orient_loss = 0

    for data, target in train_loader:
        observed_projection, expected_projection = data
        observed_projection = observed_projection.to(device)
        expected_projection = expected_projection.to(device)
        target = target.to(device)

        output = model(observed_projection, expected_projection)
        weighted_loss, loc_loss, orient_loss = loss_f(output, target)

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        batch_size = target.shape[0]
        total_weighted_loss += weighted_loss.item() * batch_size
        total_loc_loss += loc_loss.item() * batch_size
        total_orient_loss += orient_loss.item() * batch_size

    dataset_length = len(train_loader.dataset)
    avg_weighted_loss = total_weighted_loss / dataset_length
    avg_loc_loss = total_loc_loss / dataset_length
    avg_orient_loss = total_orient_loss / dataset_length

    metrics = {
        'avg_weighted_loss': avg_weighted_loss,
        'avg_loc_loss': avg_loc_loss,
        'avg_orient_loss': avg_orient_loss
    }

    return metrics


def val(model, val_loader, loss_f, device):
    model.eval()

    total_weighted_loss = 0
    total_loc_loss = 0
    total_orient_loss = 0

    with torch.no_grad():
        for data, target in val_loader:
            observed_projection, expected_projection = data
            observed_projection = observed_projection.to(device)
            expected_projection = expected_projection.to(device)
            target = target.to(device)

            output = model(observed_projection, expected_projection)
            weighted_loss, loc_loss, orient_loss = loss_f(output, target)

            batch_size = target.shape[0]
            total_weighted_loss += weighted_loss.item() * batch_size
            total_loc_loss += loc_loss.item() * batch_size
            total_orient_loss += orient_loss.item() * batch_size

    dataset_length = len(val_loader.dataset)
    avg_weighted_loss = total_weighted_loss / dataset_length
    avg_loc_loss = total_loc_loss / dataset_length
    avg_orient_loss = total_orient_loss / dataset_length

    metrics = {
        'avg_weighted_loss': avg_weighted_loss,
        'avg_loc_loss': avg_loc_loss,
        'avg_orient_loss': avg_orient_loss
    }

    return metrics


def main(config_path='./configs/config.yaml'):
    config = load_config(config_path)

    init_experiment(config)
    set_random_seed(config.seed)

    train_dataset = getattr(data, config.train.dataset.type)(config.data_root, **vars(config.train.dataset.params))
    train_loader = getattr(data, config.train.loader.type)(train_dataset, **vars(config.train.loader.params))

    val_dataset = getattr(data, config.val.dataset.type)(config.data_root, **vars(config.val.dataset.params))
    val_loader = getattr(data, config.val.loader.type)(val_dataset, **vars(config.val.loader.params))

    device = torch.device(config.device)
    model = getattr(models, config.model.type)(**vars(config.model.params)).to(device)
    optimizer = getattr(optims, config.optim.type)(model.parameters(), **vars(config.optim.params))
    scheduler = None
    loss_f = getattr(losses, config.loss.type)(**vars(config.loss.params))

    early_stopping = EarlyStopping(
        save=config.model.save,
        path=config.model.save_path,
        **vars(config.stopper.params)
    )

    train_writer = SummaryWriter(log_dir=os.path.join(config.tb_dir, 'train'))
    val_writer = SummaryWriter(log_dir=os.path.join(config.tb_dir, 'val'))

    for epoch in range(1, config.epochs + 1):
        print(f'Epoch {epoch}')
        train_metrics = train(model, optimizer, train_loader, loss_f, device)
        print_metrics('Train', train_metrics)
        write_metrics(epoch, train_metrics, train_writer)

        val_metrics = val(model, val_loader, loss_f, device)
        print_metrics('Val', val_metrics)
        write_metrics(epoch, val_metrics, val_writer)

        early_stopping(val_metrics['avg_weighted_loss'], model)  # will save the best model to disk
        if early_stopping.early_stop:
            print(f'Early stopping after {epoch} epochs.')
            break

        if scheduler:
            scheduler.step()

    train_writer.close()
    val_writer.close()

    if config.model.save:
        torch.save(model.state_dict(), config.model.save_path.replace('checkpoint', 'last_checkpoint'))


if __name__ == '__main__':
    main()
