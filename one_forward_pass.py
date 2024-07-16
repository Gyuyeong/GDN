# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.iostream import printsep

from datasets.TimeDataset import TimeDataset

from models.GDN import GDN, OutLayer
from models.graph_layer import GraphLayer

from train import train
from test import test
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores

from datetime import datetime

import argparse
import os
from pathlib import Path
import math

import random


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()


# get train and val loaders from training dataset
def get_loaders(train_dataset, seed, batch, val_ratio=0.1):
    dataset_len = int(len(train_dataset))
    train_use_len = int(dataset_len * (1 - val_ratio))
    val_use_len = int(dataset_len * val_ratio)
    val_start_index = random.randrange(train_use_len)
    indices = torch.arange(dataset_len)

    train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
    train_subset = Subset(train_dataset, train_sub_indices)

    val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
    val_subset = Subset(train_dataset, val_sub_indices)


    train_dataloader = DataLoader(train_subset, batch_size=batch,
                            shuffle=True)

    val_dataloader = DataLoader(val_subset, batch_size=batch,
                            shuffle=False)

    return train_dataloader, val_dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type = int, default=4)
    parser.add_argument('-epoch', help='train epoch', type = int, default=100)
    parser.add_argument('-slide_win', help='slide_win', type = int, default=15)
    parser.add_argument('-dim', help='dimension', type = int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=5)
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type = str, default='msl')
    parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type = int, default=0)
    parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
    parser.add_argument('-decay', help='decay', type = float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.1)
    parser.add_argument('-topk', help='topk num', type = int, default=20)
    parser.add_argument('-report', help='best / val', type = str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='')

    args = parser.parse_args()

    # set random seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
    }

    env_config={
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }

    # prepare dataset
    dataset = env_config['dataset']
    train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
    test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)

    train, test = train_orig, test_orig

    if 'attack' in train.columns:
        train = train.drop(columns=['attack'])

    feature_map = get_feature_map(dataset)  # list of features (name of columns)
    fc_struc = get_fc_graph_struc(dataset)  # fully-connected edge list

    set_device(env_config['device'])
    device = get_device()

    # create fully connected edges [2, N * (N - 1)]
    fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
    fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)

    train_dataset_indata = construct_data(train, feature_map, labels=0)
    test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())

    cfg = {
        'slide_win': train_config['slide_win'],
        'slide_stride': train_config['slide_stride'],
    }

    train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
    test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)

    train_dataloader, val_dataloader = get_loaders(train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])
    test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'], shuffle=False, num_workers=0)

    edge_index_sets = []
    edge_index_sets.append(fc_edge_index)

    # variables from config
    node_num = len(feature_map)
    embed_dim = train_config['dim']
    edge_set_num = len(edge_index_sets)
    topk = train_config['topk']
    dim = train_config['dim']
    slide_win = train_config['slide_win']
    out_layer_num = train_config['out_layer_num']
    out_layer_inter_dim = train_config['out_layer_inter_dim']

    gdn = GDN(edge_index_sets, node_num, 
              dim=dim, 
              input_dim=slide_win, 
              out_layer_num=out_layer_num, 
              out_layer_inter_dim=out_layer_inter_dim, 
              topk=topk
              ).to(device)
    
    # node embeddings
    embedding = nn.Embedding(node_num, embed_dim).to(device)
    nn.init.kaiming_uniform_(embedding.weight, a=math.sqrt(5)) # initialize node embeddings
    bn = nn.BatchNorm1d(dim).to(device)
    relu = nn.ReLU().to(device)
    leaky_relu = nn.LeakyReLU().to(device)
    dp = nn.Dropout(0.2).to(device)
    bn_outlayer_in = nn.BatchNorm1d(embed_dim).to(device)

    cache_edge_index_sets = [None] * edge_set_num
    cache_embed_index = None
    
    graph_layer = GraphLayer(in_channels=slide_win, 
                             out_channels=dim, 
                             inter_dim=dim + embed_dim, 
                             heads=1, 
                             concat=False).to(device)
    
    out_layer = OutLayer(dim * edge_set_num, node_num, out_layer_num).to(device)

    # ============================================================================================= #
    #                                 One Forward Pass of GDN                                       #
    # ============================================================================================= #
    graph_layer.eval()
    out_layer.eval()
    with torch.no_grad():
        for x, labels, attack_labels, edge_index in train_dataloader:
            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

            batch_num, node_num, all_feature = x.shape  # [# batch, # nodes, # features]
            x = x.view(-1, all_feature).contiguous()

            gcn_outs = []
            for i, edge_index in enumerate(edge_index_sets):
                edge_num = edge_index.shape[1]
                cache_edge_index = cache_edge_index_sets[i]

                if cache_edge_index is None or cache_edge_index.shape[1] != edge_num * batch_num:
                    cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)

                all_embeddings = embedding(torch.arange(node_num).to(device)).to(device)  # [# nodes, embed_dim]

                weights_arr = all_embeddings.detach().clone()
                all_embeddings = all_embeddings.repeat(batch_num, 1)  # [# nodes * # batch, embed_dim]

                weights = weights_arr.view(node_num, -1)  # [# nodes, embed_dim]

                cos_ji_mat = torch.matmul(weights, weights.T)  # [# nodes, # nodes]
                normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))  # [# nodes, # nodes]
                cos_ji_mat /= normed_mat

                topk_indices_ji = torch.topk(cos_ji_mat, topk, dim=-1).indices  # indices [# nodes, topk]

                learned_graph = topk_indices_ji

                gated_i = torch.arange(0, node_num).unsqueeze(1).repeat(1, topk).flatten().to(device).unsqueeze(0)  # [1, # node * topk] [0, 0, ... 0, 1, 1, ... 1, ... , N-1, N-1, ... N-1]
                gated_j = topk_indices_ji.flatten().unsqueeze(0)  # [1, # node * topk]
                gated_edge_index = torch.cat((gated_i, gated_j), dim=0)  # [2, # node * topk]

                batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)  # [2, # node * topk * # batch]

                out, (new_edge_index, att_weight) = graph_layer(x, batch_gated_edge_index, all_embeddings, return_attention_weights=True)  # [# node * # batch, embed_dim]

                out = bn(out)
                out = relu(out)

                gcn_outs.append(out)

                print("out shape =", out.shape)
                print("new_edge_index shape =", new_edge_index.shape)  # [2, # node * # batch * topk]
                print("att_weight shape =", att_weight.shape)  # [# node * # batch * topk, 1, 1]

            x = torch.cat(gcn_outs, dim=1)
            print(x.shape)
            x = x.view(batch_num, node_num, -1).to(device)  # [# batch, # node, embed_dim]
            print(x.shape)

            indexes = torch.arange(0, node_num).to(device)
            out = torch.mul(x, embedding(indexes))
            print(out.shape)

            out = out.permute(0, 2, 1)
            out = F.relu(bn_outlayer_in(out))
            out = out.permute(0, 2, 1)

            print(out.shape)  # [# batch, # node, embed_dim]

            out = dp(out)
            out = out_layer(out)
            print(out.shape)  # [# batch, # node, 1]
            out = out.view(-1, node_num)
            print(out.shape)  # [# batch, # node]
            break
