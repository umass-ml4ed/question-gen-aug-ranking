# Imports 
import os 
import wandb
import copy
import math
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
import numpy as np 
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup

from ranking_kl.bert_rank import get_transformer_input, tokenize, get_dataloader, RougeKLModel


def get_score(model, device, test_dataloader, no_token_type=False):
    all_scores = []
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        if no_token_type:
            b_inp_ids, b_attn_masks = batch
            b_inp_ids, b_attn_masks = b_inp_ids.to(device), b_attn_masks.to(device)
        else:
            b_inp_ids, b_token_ids, b_attn_masks = batch
            b_inp_ids, b_token_ids, b_attn_masks = b_inp_ids.to(device), b_token_ids.to(device), b_attn_masks.to(device)

        with torch.no_grad():
            if no_token_type:
                scores = model(b_inp_ids, b_attn_masks)
            else:
                scores = model(b_inp_ids, b_attn_masks, b_token_ids)
            all_scores.extend(scores.tolist())
    return all_scores

def get_top_1_df(test_df):
    stry_name, content, attr, ex_or_im, loc_or_sum, ac_ques, gen_ques = [], [], [], [], [], [], []
    pair_id, grp_wise_max_score = [], []
    grp_pair_ids = test_df.groupby('pairID')
    for grp_full in grp_pair_ids:
        grp = grp_full[1]
        max_score = max(grp['Score'])
        max_index = grp['Score'].tolist().index(max_score)
        # print('max_score:', max_score)
        pair_id.append(grp['pairID'].tolist()[0])
        stry_name.append(grp['story_name'].tolist()[0])
        content.append(grp['content'].tolist()[0])
        grp_wise_max_score.append(max_score)
        ac_ques.append(grp['question'].tolist()[0])
        gen_ques.append(grp['generated_question'].iloc[max_index])
        attr.append(grp['attribute'].tolist()[0])
        ex_or_im.append(grp['ex_or_im'].tolist()[0])
        loc_or_sum.append(grp['local_or_sum'].tolist()[0])
    
    reduced_df = pd.DataFrame()
    reduced_df['pairID'] = pair_id
    reduced_df['story_name'] = stry_name
    reduced_df['content'] = content
    reduced_df['attribute'] = attr
    reduced_df['local_or_sum'] = loc_or_sum
    reduced_df['ex_or_im'] = ex_or_im
    reduced_df['question'] = ac_ques
    reduced_df['generated_question'] = gen_ques
    reduced_df['Score'] = grp_wise_max_score
    return reduced_df

def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-NTT', '--no_token_type', action=argparse.BooleanOptionalAction, help='If model does not have token type ids')
    parser.add_argument("-MN", "--model_name", type=str, default="bert-base-uncased", help="Variant of the Transformer model for finetuning")
    parser.add_argument("-SN", "--save_name", type=str, default="bert_org_10_0.1_0.1", help="Model name for saving")
    parser.add_argument('-TFN', '--test_file_name', type=str, default="nucleus_flan_t5_large_org_16_sa_0.90_1.00_10.csv", help="Test File name")
    parser.add_argument('-Attr', '--use_attr', action=argparse.BooleanOptionalAction, help='Use attribute')
    parser.add_argument('-ExIm', '--use_ex_im', action=argparse.BooleanOptionalAction, help='Use explicit implict tag')
    parser.add_argument("-ML", "--max_len", type=int, default=512, help="max length for tokenizers")
    parser.add_argument("-PC", "--perfix_choice", type=int, default=1, help="Style of input construction to the model")
    parser.add_argument("-B", "--batch_size", type=int, default=10, help="Batch size (should be equal to the number of questions generated)")
    params = parser.parse_args()
    return params

def main():
    # parse arguments
    args = add_params()

    # Read data 
    test_dir = './data/results_org/'
    test_df = pd.read_csv(os.path.join(test_dir, args.test_file_name))

    # get input
    test_context_ans, test_question = get_transformer_input(test_df, choice=args.perfix_choice, use_attr=args.use_attr, use_ex_im=args.use_ex_im)

    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    test_tokenized_dict = tokenize(tokenizer, test_context_ans, test_question, max_len=args.max_len)
    print('Tokenized the input!')

    # get data loader 
    batch_size = args.batch_size # should be equal to the number of questions
    test_dataloader = get_dataloader(batch_size, test_tokenized_dict, rouge_scores=None, no_token_type=args.no_token_type)

    # initialze device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
 
    # Load the model 
    save_dir = './ranking_kl/Checkpoints_sep'
    save_path = os.path.join(save_dir, args.save_name)
    model = torch.load(save_path).to(device)

    print('Successfully loaded the model!')
    all_scores = get_score(model, device, test_dataloader, args.no_token_type)

    # save into df
    test_df['Score'] = all_scores
    results_dir = './data/results_rank_kl_metadata'
    test_df.to_csv(os.path.join(results_dir, args.save_name + '_' + args.test_file_name), index=False)

    # TODO: choose top 1 per pair id
    reduced_df = get_top_1_df(test_df)
    results_dir = './data/results_rank_kl'
    reduced_df.to_csv(os.path.join(results_dir, args.save_name + '_' + args.test_file_name), index=False)


if __name__ == '__main__':
    main()