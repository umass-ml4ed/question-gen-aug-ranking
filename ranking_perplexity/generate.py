import torch
import torch.backends.cudnn as cudnn
import argparse
import os
import json
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import pathlib
import time
from transformers import T5Tokenizer, BartTokenizer

from finetune.finetune_org import FinetuneTransformer
from ranking_perplexity.rank import rank
from finetune.inference_org import get_parallel_corpus, construct_transformer_input_old_vary, get_transformer_encoding, FairyDataset, get_dataloader
from utils.handle_data import load_df, RAW_DIR, save_csv


def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("-MT", "--model_type", type=str, default="t", help="T for T5 and B for BART")
    parser.add_argument("-MN", "--model_name", default="t5-small", help="Variant of the Transformer model for finetuning")
    parser.add_argument("-N", "--run_name", type=str, default="flan_t5_large_aug_0.8", help="Name of the Run (Used in storing the model)")
    parser.add_argument("-EFN", "--eval_filename", type=str, default="test.json", help="Evaluation filename")
    parser.add_argument("-CF", "--checkpoint_folder", type=str, default="Checkpoints_org", help="Folder where the checkpoint is stored")
    parser.add_argument("-B", "--batch_size", type=int, default=8, help="Batch size for passing through the Transformer Model")

    parser.add_argument('-DS', '--decoding_strategies', type=str, default="G", help='Specify the decoding strategy (B-Beam Search, N-Nucleus sampling, C - Contrsative, G-Greedy)')
    parser.add_argument("-PS", "--p_sampling", type=float, default=0.9, help="Value of P used in the P-sampling")
    parser.add_argument("-T", "--temperature", type=float, default=1, help="Temperature for softmax decoding")
    parser.add_argument("-K", "--top_K", type=int, default=4, help="Value of K used for contrastive decoding")
    parser.add_argument("-alpha", "--alpha", type=float, default=0.6, help="Value of alpha used for contrastive decoding")
    parser.add_argument("-NS", "--num_of_samples", type=int, default=10, help="Number of samples to generate when using sampling")
    parser.add_argument('-NB', '--num_of_beams', type=int, default=3, help="Number of beams for decoding")
    
    parser.add_argument('-D', '--debug', action='store_true', help='Debug mode evaluating on a small subset of 5 samples')
    parser.add_argument('-S', '--seeds', default="37", type=str, help='Random seed') 
    parser.add_argument('-TO', '--top_one', action=argparse.BooleanOptionalAction, help='Get top one based on perplexity')
    
    params = parser.parse_args()
    
    return params

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True

def load_data(args, test_data, tokenizer):
    # Prepare dataloader
    test_story, test_answer, test_question = get_parallel_corpus(test_data)
    test_inps = construct_transformer_input_old_vary(test_story, test_answer)
    test_input_ids, test_attention_mask, test_labels = get_transformer_encoding(tokenizer, test_inps, test_question)
    test_dataset = FairyDataset(test_input_ids, test_attention_mask, test_labels)
    test_dataloader = get_dataloader(args.batch_size, test_dataset, datatype='val')
    return test_dataloader

def load_model(args, device):
    search_dir = os.path.join('./finetune', args.checkpoint_folder, args.run_name)
    for file in os.listdir(search_dir):
        name, ext = os.path.splitext(file)
        if ext == '.ckpt':
            ckpt_file = os.path.join(search_dir, file)
    print('ckpt_file', ckpt_file)
    model = FinetuneTransformer.load_from_checkpoint(ckpt_file, model_type = args.model_type).model.to(device)
    print('Successfully loaded the saved checkpoint!')
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model.eval()

    return model, tokenizer

def compute_perplexity(logits, labels):
    """
    Compute the perplexity using logits (dimension = (seq_len, vocab_size) 
    and labels (dimension = (seq_len))
    """
    return torch.exp(torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), reduction='mean'))


def generate(device, model, val_dataloader, force_words_ids, decoding_strategy, num_beams=10, prob_p=0.9, temp=1, K=4, alpha=0.6, num_samples=10):
    val_outputs = []
    val_outputs_ppl = []
    for batch in tqdm(val_dataloader):
        val_input_ids = batch['input_ids'].to(device)
        # TODO: Force ? to occur in the sentence
        if decoding_strategy == 'B': # Beam search
            generation = model.generate(val_input_ids, force_words_ids=force_words_ids, 
                                        num_beams = num_beams, temperature=temp,
                                        num_return_sequences=num_samples, 
                                        max_new_tokens=64, 
                                        output_scores=True, return_dict_in_generate=True)
        elif decoding_strategy == 'N': # Nucleus Sampling
            generation = model.generate(val_input_ids, do_sample=True, max_new_tokens=64,
                                        top_p=prob_p, temperature=temp,
                                        num_return_sequences=num_samples, 
                                        output_scores=True, return_dict_in_generate=True)
        elif decoding_strategy == 'C': # Contrastive Decoding
            generation = model.generate(val_input_ids, do_sample=True, max_new_tokens=64,
                                        penalty_alpha=alpha, top_k=K,
                                        num_return_sequences=num_samples, 
                                        output_scores=True, return_dict_in_generate=True)
            
        for idx in range(generation['sequences'].shape[0]):
            gen = generation['sequences'][idx]
            valid_gen_idx = torch.where(gen!=0)[0]
            logits = torch.vstack([generation['scores'][i][idx].unsqueeze(0) for i in valid_gen_idx-1])
            ppl = compute_perplexity(logits, gen[gen!=0])
            assert(torch.isnan(ppl) == False)
            val_outputs.append(gen)
            val_outputs_ppl.append(ppl.item())

    return val_outputs, val_outputs_ppl

def get_preds(tokenizer, generated_tokens):
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    val_preds = []
    for inp in generated_tokens:
        sample = tokenizer.decode(inp, skip_special_tokens=True)
        val_preds.append(sample)
    return val_preds

def generate_wrapper(model, tokenizer, val_dataloader, val_df, args, device):
    force_tokens = ['?']
    force_words_ids = tokenizer(force_tokens, add_special_tokens=False).input_ids
    
    all_df = []
    for decoding_strategy in args.decoding_strategies.split('-'):
        for seed in args.seeds.split('-'):
            val_df_curr = val_df.copy()
            set_random_seed(int(seed))
            val_outputs, val_outputs_ppl = generate(device, model, val_dataloader, force_words_ids, 
                                                    decoding_strategy,
                                                    args.num_of_beams, 
                                                    args.p_sampling, args.temperature, 
                                                    args.top_K, args.alpha,
                                                    args.num_of_samples)
            val_preds = get_preds(tokenizer, val_outputs)

            val_preds = [val_preds[x:x+args.num_of_samples] for x in range(0, len(val_preds), args.num_of_samples)]
            val_outputs_ppl = [val_outputs_ppl[x:x+args.num_of_samples] for x in range(0, len(val_outputs_ppl), args.num_of_samples)]
            #print(val_preds)
            #print(val_outputs_ppl)
            val_df_curr['generated_question'] = val_preds
            val_df_curr['score'] = val_outputs_ppl
            all_df.append(val_df_curr)

    all_df = pd.concat(all_df)
    return all_df

def main():
    set_random_seed(seed = 37)
    args = add_params() 

    test_file = os.path.join('./data', args.eval_filename)

    # Load test data
    test_data = []
    with open(test_file, 'r') as infile:
        for i, line in enumerate(infile):
            if args.debug and i > 5:
                break # choose only 5 in the case of debug
            json_dict = json.loads(line)
            json_dict['pairID'] = i+1
            test_data.append(json_dict)
    test_df = pd.DataFrame(test_data)

    output_path = os.path.join(RAW_DIR, "results_rank")
    if args.decoding_strategies == 'N':
        save_csv_name = 'nucleus_{:s}_{:.2f}_{:.2f}_{:d}'.format(args.run_name, args.p_sampling, args.temperature, args.num_of_samples)
    elif args.decoding_strategies == 'C':
        save_csv_name = 'contrastive_{:s}_{:d}_{:.2f}_{:d}'.format(args.run_name, args.top_K, args.alpha, args.num_of_samples)
    else:
        save_csv_name = args.run_name
    
    save_csv_path = os.path.join(output_path, save_csv_name + '.csv')
    print(save_csv_path)

    # check if results file exists
    if os.path.exists(save_csv_path):
        test_df = pd.read_csv(save_csv_path)
        print('Output file already exists')
    else: # else produce results
        # Load model type
        if args.model_type == 'T':
            tokenizer = T5Tokenizer.from_pretrained(args.model_name)
        elif args.model_type == 'B':
            tokenizer = BartTokenizer.from_pretrained(args.model_name)
        else:
            print('Wrong model type - either T or B only')

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
        model, tokenizer = load_model(args, device)
        test_dataloader = load_data(args, test_data, tokenizer)
        test_df = generate_wrapper(model, tokenizer, test_dataloader, test_df, args, device)

    # Get top-k generated questions according to scores for each pair id
    df_ranked, df_test = rank(test_df, top_one = args.top_one)
    # Save top-k generated questions for each pair id in submission format
    if args.top_one:
        save_csv_name = save_csv_name + '_top_1'
    save_csv(df_ranked, save_csv_name, output_path)


if __name__ == '__main__':
    main()