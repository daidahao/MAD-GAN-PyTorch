import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_curve, roc_auc_score

from data import read_data
from model import MADGAN

def train(model: MADGAN, data, epoch, train_lr, batch_size):
    g_optimizer = torch.optim.AdamW(model.generator.parameters() , lr=train_lr)
    d_optimizer = torch.optim.AdamW(model.discriminator.parameters() , lr=train_lr)

    dataset = TensorDataset(data, data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    bcel = nn.BCELoss(reduction = 'mean')

    gls, dls = [], []
    for real_data, _ in dataloader:
        # labels for discriminator
        batch_size = real_data.shape[0]
        real_labels = 0.9 * torch.ones(batch_size, 1, dtype=torch.double) # label smoothing
        fake_labels = 0.1 * torch.ones(batch_size, 1, dtype=torch.double)

        # training discriminator
        d_optimizer.zero_grad()
        real_output = model.discriminator(real_data)
        real_loss = bcel(real_output, real_labels)

        noise = torch.randn(batch_size, model.n_window, model.n_latent, dtype=torch.double)
        fake_data = model.generator(noise)
        fake_output = model.discriminator(fake_data.detach())
        fake_loss = bcel(fake_output, fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()

        # training generator
        g_optimizer.zero_grad()
        fake_output = model.discriminator(fake_data)
        g_loss = bcel(fake_output, real_labels)
        g_loss.backward()
        g_optimizer.step()
        
        gls.append(g_loss.item()); dls.append(d_loss.item())
    tqdm.write(f'Epoch {epoch},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
    return np.mean(gls) + np.mean(dls)

def test(model: MADGAN, data, n_iters, train_lr, batch_size):
    dataset = TensorDataset(data, data)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    msel = nn.MSELoss(reduction='none')

    scores = []
    for real_data, _ in dataloader:
        # reverse data to latent space
        batch_size = real_data.shape[0]
        noise = torch.randn(batch_size, model.n_window, model.n_latent, \
            dtype=torch.double, requires_grad=True)
        optimizer = torch.optim.AdamW([noise], lr=train_lr, weight_decay=1e-5)
        for _ in range(n_iters):
            optimizer.zero_grad()
            fake_data = model.generator(noise)
            loss = msel(fake_data, real_data)
            torch.mean(loss).backward()
            optimizer.step()
        loss = msel(fake_data[:, -1, :], real_data[:, -1, :])
        scores.append(loss.detach().numpy())
    scores = np.vstack(scores)
    return scores

if __name__ == '__main__':

    ## Parse arguments
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--train_data', type=str, default='data/SWaT_Dataset_Normal_v0.csv')
    parser.add_argument('--test_data', type=str, default='data/SWaT_Dataset_Attack_v0.csv')
    parser.add_argument('--train_skiprows', type=int, default=60*60*10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--time_column', type=int, default=0)
    parser.add_argument('--label_column', type=int, default=-2)
    parser.add_argument('--sampling_period', type=int, default=10)
    parser.add_argument('--n_window', type=int, default=5)
    # training arguments
    parser.add_argument('--train_lr', type=float, default=1e-4)
    parser.add_argument('--train_epochs', type=int, default=100)
    # testing arguments
    parser.add_argument('--test_iters', type=int, default=10)
    # model arguments
    parser.add_argument('--model_path', type=str, default='model.pt')
    # get arguments
    args = parser.parse_args()

    t_start = timer()

    ## Prepare data
    data, ts, _, normalizer = read_data(args.train_data, args.train_skiprows, 
        args.sampling_period, args.time_column, args.label_column, None, args.n_window)
    n_feats = data.shape[2]
    print(f'Number of features: {n_feats}, number of samples: {data.shape[0]}')
    print(f'Window size: {args.n_window}, sampling period: {args.sampling_period} seconds')
    
    ## Init model
    model = MADGAN(n_feats, args.n_window).double()
    
    ## Train model
    # Load model if file exists
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))
        print(f"Loading model from {args.model_path}")
    else:
    # Otherwise, train from scratch
        print(f"No model found at {args.model_path}")
        print(f"Creating new model: {model.name}")

        for epoch in tqdm(range(0, args.train_epochs)):
            _ = train(model, data, epoch, args.train_lr, args.batch_size)
        print(f'Training time: {timer() - t_start:.2f} seconds')

    ## Test
    torch.zero_grad = True
    model.eval()
    
    print(f'Testing')
    t_start = timer()

    # Load test data
    data, ts, labels, _ = read_data(args.test_data, 0,
        args.sampling_period, args.time_column, args.label_column, normalizer, args.n_window)
    print(f'Number of features: {n_feats}, number of samples: {data.shape[0]}')
    print(f'Window size: {args.n_window}, sampling period: {args.sampling_period} seconds')

    # Test
    scores = test(model, data, args.test_iters, args.train_lr, args.batch_size)
    scores = np.mean(scores, axis=1)
    print(f'Testing time: {timer() - t_start:.2f} seconds')

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    argmax = np.argmax(f1)
    auc_roc = roc_auc_score(labels, scores)
    print(f'Precision: {precision[argmax]:.3f}, Recall: {recall[argmax]:.3f}, F1: {f1[argmax]:.3f}, Threshold: {thresholds[argmax]:.3f}')
    print(f'AUC-ROC: {auc_roc:.3f}')

    # Save model
    print(f'Saving model to {args.model_path}')
    torch.save(model.state_dict(), args.model_path)
