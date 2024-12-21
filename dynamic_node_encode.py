import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model_list import MODEL
from preprocess import load_dataset, TemporalDataset
from utils import initialize_seed, get_name

dim = "dim_128"
args_config = 'saved_models/SZ_TAXI_D/SPASPOSSearch_12_43_all.json'

def get_args():
    with open(args_config, 'r') as f:
        args = json.load(f)
    return argparse.Namespace(**args)   # 将字典转换为 Namespace

def poi_emb(train_loader, model):
    node_emb, node_cnt = None, 156
    for timestamps in train_loader:
        g_batched_list, time_batched_list = model.get_batch_graph_list(timestamps, model.train_seq_len,
                                                                       model.dataset_graph, split="train")
        hist_embeddings, start_time_tensor, hist_embeddings_transformer, attn_mask = model.pre_forward(g_batched_list,
                                                                                                       time_batched_list)
        train_graphs, time_batched_list_t = g_batched_list[-1], time_batched_list[-1]
        prev_graph_embeds_list, time_diff_tensor, prev_graph_embeds_transformer_list, local_attn_mask = model.get_prev_embeddings(
            train_graphs,
            hist_embeddings,
            start_time_tensor,
            model.train_seq_len - 1, hist_embeddings_transformer, attn_mask)
        node_sizes = [len(g.nodes()) for g in train_graphs]
        _, per_graph_ent_embeds = model.get_per_graph_ent_embeds(train_graphs, time_batched_list_t, node_sizes,
                                                                 time_diff_tensor, prev_graph_embeds_list,
                                                                 prev_graph_embeds_transformer_list, local_attn_mask,
                                                                 full=False)
        t, g, ent_embed = time_batched_list_t[0], train_graphs[0], per_graph_ent_embeds[0]
        # print(train_graphs[0].edges())
        node_emb = ent_embed[-node_cnt:]
    pd.DataFrame(node_emb.detach().cpu().numpy()).to_csv('./emb/poi/{}.csv'.format(dim), header=False, index=False)

def weather_emb(train_loader, model):
    time_emb = []
    for timestamps in train_loader:
        g_batched_list, time_batched_list = model.get_batch_graph_list(timestamps, model.train_seq_len,
                                                                       model.dataset_graph, split="train")
        hist_embeddings, start_time_tensor, hist_embeddings_transformer, attn_mask = model.pre_forward(g_batched_list,
                                                                                                       time_batched_list)
        train_graphs, time_batched_list_t = g_batched_list[-1], time_batched_list[-1]
        prev_graph_embeds_list, time_diff_tensor, prev_graph_embeds_transformer_list, local_attn_mask = model.get_prev_embeddings(
            train_graphs,
            hist_embeddings,
            start_time_tensor,
            model.train_seq_len - 1, hist_embeddings_transformer, attn_mask)
        node_sizes = [len(g.nodes()) for g in train_graphs]
        _, per_graph_ent_embeds = model.get_per_graph_ent_embeds(train_graphs, time_batched_list_t, node_sizes,
                                                                 time_diff_tensor, prev_graph_embeds_list,
                                                                 prev_graph_embeds_transformer_list, local_attn_mask,
                                                                 full=False)
        t, g, ent_embed = time_batched_list_t[0], train_graphs[0], per_graph_ent_embeds[0]

        lst = list(map(int, torch.cat(train_graphs[0].edges(), dim=0)))
        lst.sort()
        node_id = lst[len(lst) // 2]
        print(timestamps, train_graphs[0].edges(), node_id)
        time_emb.append(list(ent_embed[node_id]))
    print(np.array(time_emb).shape)
    pd.DataFrame(np.array(time_emb)).to_csv('./emb/weather/{}.csv'.format(dim), header=False, index=False)

def main():
    args = get_args()

    dataset_info_dict = load_dataset(args.dataset_dir + args.dataset)
    train_dataset = TemporalDataset(dataset_info_dict['train_timestamps'], toy=args.sampled_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, num_workers=0)
    initialize_seed(args.random_seed)

    model = MODEL[args.encoder](args, dataset_info_dict, 'cuda:0')
    model.load_state_dict(torch.load(f'{args.saved_model_dir}{get_name(args)}.pth'))
    model.ent_encoder.ops = args.arch
    model = model.cuda()
    # poi_emb(train_loader, model)
    weather_emb(train_loader, model)

if __name__ == '__main__':
    main()

