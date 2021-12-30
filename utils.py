import numpy as np
import pickle
import torch
import wandb

from collections import defaultdict
from matplotlib import pyplot as plt
from torch import optim
from torchvision.utils import make_grid
from tqdm.notebook import tqdm


def train_epoch(model, train_loader, optimizer, use_cuda, loss_key='total'):
    model.train()
  
    stats = defaultdict(list)
    for x in train_loader:
        if use_cuda:
            x = x.cuda()
        losses = model.loss(x)
        optimizer.zero_grad()
        losses[loss_key].backward()
        optimizer.step()

        for k, v in losses.items():
            stats[k].append(v.item())
        
        for k, v in losses.items():
            wandb.log({f'{k}_train': v.item()})

    return stats


def eval_model(model, data_loader, use_cuda):
    model.eval()
    stats = defaultdict(float)
    with torch.no_grad():
        for x in data_loader:
            if use_cuda:
                x = x.cuda()
            losses = model.loss(x)
            for k, v in losses.items():
                stats[k] += v.item() * x.shape[0]

        for k in stats.keys():
            stats[k] /= len(data_loader.dataset)
    return stats


def train_model(model, 
                train_loader, 
                test_loader, 
                epochs, 
                lr, 
                noise=None, 
                preprocess=lambda x: x,
                n_samples=10, 
                use_tqdm=False, 
                use_cuda=False, 
                loss_key='total_loss'
                ):

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = defaultdict(list)
    test_losses = defaultdict(list)
    forrange = tqdm(range(epochs)) if use_tqdm else range(epochs)
    if use_cuda:
        model = model.cuda()
        
    for epoch in forrange:
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, use_cuda, loss_key)
        test_loss = eval_model(model, test_loader, use_cuda)

        if n_samples > 0 or core is not None:
            with torch.no_grad():
                if noise is not None:
                    samples = model.sample(noise=noise)
                else:
                    samples = model.sample(n_samples)
            images = wandb.Image(preprocess(samples))
            wandb.log({"Samples": images})

        for k in test_loss.keys():
            wandb.log({f'{k}_test': test_loss[k]})

        for k in train_loss.keys():
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])
    return dict(train_losses), dict(test_losses)



def plot_training_curves(train_losses, test_losses, logscale_y=False, logscale_x=False):
    n_train = len(train_losses[list(train_losses.keys())[0]])
    n_test = len(test_losses[list(train_losses.keys())[0]])
    x_train = np.linspace(0, n_test - 1, n_train)
    x_test = np.arange(n_test)

    plt.figure()
    for key, value in train_losses.items():
        plt.plot(x_train, value, label=key + '_train')

    for key, value in test_losses.items():
        plt.plot(x_test, value, label=key + '_test')

    if logscale_y:
        plt.semilogy()
    
    if logscale_x:
        plt.semilogx()

    plt.legend(fontsize=12)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.show()


def load_pickle(path, flatten=True):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    train_data, test_data = data['train'], data['test']
    train_data = np.transpose(train_data.astype('uint8'), (0, 3, 1, 2))
    test_data = np.transpose(test_data.astype('uint8'), (0, 3, 1, 2))
    if flatten:
        train_data = train_data.reshape(-1, 28 * 28)
        test_data = test_data.reshape(-1, 28 * 28)
    return train_data, test_data

def show_samples(samples, title, preprocess=lambda x: x):
    grid_img = preprocess(samples)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img)
    plt.axis('off')
    plt.show()


def visualize_images(data, title):
    idxs = np.random.choice(len(data), replace=False, size=(100,))
    images = data[idxs]
    show_samples(images, title, preprocess=grid_preprocessing)

def grid_preprocessing(samples):
    grid_samples = make_grid(samples, nrow = int(np.sqrt(len(samples))))
    grid_samples = grid_samples.permute(1, 2, 0) 
    return grid_samples.numpy()