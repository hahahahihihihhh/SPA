import argparse
import json

import torch
from torch.utils.data import DataLoader

from model_list import MODEL
from preprocess import load_dataset, TemporalDataset
from utils import initialize_seed, get_name

args_config = 'saved_models/SPASPOSSearch_8_1_all.json'

def get_args():
    with open(args_config, 'r') as f:
        args = json.load(f)
    return argparse.Namespace(**args)   # 将字典转换为 Namespace

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

    for timestamps in train_loader:
        print(timestamps)

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
        print(t, g, ent_embed)


if __name__ == '__main__':
    main()

