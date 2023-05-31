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

def append_rouge_score(df):
    '''
    Utility function to get rouge scores
    '''
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    all_scores = []
    for i, row in df.iterrows():
        score = scorer.score(str(row['question']), str(row['generated_question']))
        all_scores.append(score['rougeL'].fmeasure)
    df['Rouge_L'] = all_scores

def get_transformer_input(df, choice=1, use_attr=False, use_ex_im=False):
    context_ans, question = [], []
    for i, row in df.iterrows():
        if choice == 1:
            if use_ex_im and use_attr:
                context_ans_str = 'Ex_Im: {:s}\tAttribute: {:s}\tContext: {:s}\tAnswer: {:s}'.format(row['ex_or_im'], row['attribute'], row['content'], row['answer'])
            elif use_ex_im:
                context_ans_str = 'Ex_Im: {:s}\tContext: {:s}\tAnswer: {:s}'.format(row['ex_or_im'], row['content'], row['answer'])
            elif use_attr:
                context_ans_str = 'Attribute: {:s}\tContext: {:s}\tAnswer: {:s}'.format(row['attribute'], row['content'], row['answer'])
            else:
                context_ans_str = 'Context: {:s}\tAnswer: {:s}'.format(row['content'], row['answer'])
        elif choice == 2:
            if use_ex_im and use_attr:
                context_ans_str = 'Attribute: {:s}\tEx_Im: {:s}\tAnswer: {:s}\tContext: {:s}'.format(row['attribute'], row['ex_or_im'], row['answer'], row['content'])
            elif use_ex_im:
                context_ans_str = 'Ex_Im: {:s}\tAnswer: {:s}\tContext: {:s}'.format(row['ex_or_im'], row['answer'], row['content'])
            elif use_attr:
                context_ans_str = 'Attribute: {:s}\tAnswer: {:s}\tContext: {:s}'.format(row['attribute'], row['answer'], row['content'])
            else:
                context_ans_str = 'Answer: {:s}\tContext: {:s}'.format(row['answer'], row['content'])
        context_ans.append(context_ans_str)
        question.append(row['generated_question'])
    return context_ans, question

def tokenize(tokenizer, context_ans, question, max_len=512):
    tokenize_input = [(ca, q) for ca, q in zip(context_ans, question)]
    encode_dict = tokenizer.batch_encode_plus(tokenize_input, 
                                              # pad_to_max_length = True,
                                              max_length = max_len, 
                                              padding = 'longest',
                                              truncation = 'only_first',
                                              return_tensors='pt',
                                              )
    return encode_dict

def get_dataloader(batch_size, tokenize_dict, rouge_scores=None, no_token_type=False):
    if no_token_type:
        if rouge_scores is None:
            data = TensorDataset(tokenize_dict['input_ids'], tokenize_dict['attention_mask'])
        else:
            rouge_tensor = torch.tensor(rouge_scores, dtype=torch.float32)
            data = TensorDataset(tokenize_dict['input_ids'], tokenize_dict['attention_mask'], rouge_tensor)
    else:
        if rouge_scores is None:
            data = TensorDataset(tokenize_dict['input_ids'], tokenize_dict['token_type_ids'],
                                tokenize_dict['attention_mask'])
        else:
            rouge_tensor = torch.tensor(rouge_scores, dtype=torch.float32)
            data = TensorDataset(tokenize_dict['input_ids'], tokenize_dict['token_type_ids'],
                                tokenize_dict['attention_mask'], rouge_tensor)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader

class RougeKLModel(nn.Module):
    def __init__(self, model_name, hid_dim = 768) -> None:
        super().__init__()
        self.trans_model = AutoModel.from_pretrained(model_name)
        # define the regression head
        self.linear = nn.Linear(hid_dim, 1)
    
    def forward(self, input_ids, attn_masks, token_ids = None):
        if token_ids is None:
            trans_out = self.trans_model(input_ids = input_ids, attention_mask = attn_masks)
        else:
            trans_out = self.trans_model(input_ids = input_ids, token_type_ids = token_ids, attention_mask = attn_masks)
        trans_cls = trans_out['last_hidden_state'][:, 0, :]
        score = self.linear(trans_cls).squeeze(1)
        return score

def get_optimizer_scheduler(model, lr, train_dataloader_len, epochs):
    optimizer = AdamW(model.parameters(),
                lr = lr, # args.learning_rate 
                eps = 1e-8 # args.adam_epsilon
            )
    total_steps = train_dataloader_len * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    return optimizer, scheduler

def train(use_wandb, no_token_type, epochs, model, device, train_dataloader, val_dataloader, optimizer, scheduler, alpha1, alpha2, save_path):
    epochswise_train_losses, epochwise_val_losses = [], []
    prev_val_loss, early_stop_ctr, early_stop_threshold = 0, 0, 3
    least_val_loss, cur_least_epoch = math.inf, 0

    kl_loss = nn.KLDivLoss(reduction="batchmean")

    # Start training 
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Reset the total loss for this epoch.
        total_loss = 0

        # Switch model to the train mode
        model.train()

        # Iterate over all batches
        for step, batch in tqdm(enumerate(train_dataloader), total = len(train_dataloader)):
            if no_token_type:
                b_inpids, b_attnids, b_rs = batch
                b_inpids, b_attnids, b_rs = b_inpids.to(device), b_attnids.to(device), b_rs.to(device)
            else:
                b_inpids, b_ttids, b_attnids, b_rs = batch
                b_inpids, b_ttids, b_attnids, b_rs = b_inpids.to(device), b_ttids.to(device), b_attnids.to(device), b_rs.to(device)

            # Clear previously accumulated gradients
            model.zero_grad()        

            # Forward pass
            if no_token_type:
                scores = model(b_inpids, b_attnids)
            else:
                scores = model(b_inpids, b_attnids, b_ttids)

            # NOTE: KL Loss Calculation
            scores_log_softmax = F.log_softmax(alpha1 * scores, dim=0)
            rouge_softmax = F.softmax(alpha2 * b_rs, dim=0)
            kl_loss_batch = kl_loss(scores_log_softmax, rouge_softmax) # divide by graident accumulation
            if step % 1000 == 0:
                print(f'Step {step} loss: {kl_loss_batch}')
            # import pdb; pdb.set_trace()

            # NOTE: Implement Graident accumulation 
            # https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation
            total_loss += kl_loss_batch.item()
            kl_loss_batch.backward()
            # Parameter Update
            optimizer.step() 
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        epochswise_train_losses.append(avg_train_loss) 
        print("Average training loss: {0:.6f}".format(avg_train_loss))

        # NOTE: Validation 
        tot_val_loss = 0 
        for val_step, val_batch in tqdm(enumerate(val_dataloader)):
            if no_token_type:
                b_inpids_val, b_attnids_val, b_rs_val = val_batch
                b_inpids_val, b_attnids_val, b_rs_val = b_inpids_val.to(device), b_attnids_val.to(device), b_rs_val.to(device)
            else:
                b_inpids_val, b_ttids_val, b_attnids_val, b_rs_val = val_batch
                b_inpids_val, b_ttids_val, b_attnids_val, b_rs_val = b_inpids_val.to(device), b_ttids_val.to(device), b_attnids_val.to(device), b_rs_val.to(device)

            with torch.no_grad():
                if no_token_type:
                    valscores = model(b_inpids_val, b_attnids_val)
                else:
                    valscores = model(b_inpids_val, b_attnids_val, b_ttids_val)
                # NOTE: KL Loss Calculation
                scores_log_softmax_val = F.log_softmax(alpha1 * valscores, dim=0)
                rouge_softmax_val = F.softmax(alpha2 * b_rs_val, dim=0)
                kl_loss_batch_val = kl_loss(scores_log_softmax_val, rouge_softmax_val)
                tot_val_loss += kl_loss_batch_val
            
        avg_val_loss = tot_val_loss / len(val_dataloader)
        print(" Average validation loss: {0:.6f}".format(avg_val_loss))
        epochwise_val_losses.append(avg_val_loss)
        
        if avg_val_loss < least_val_loss:
            cur_least_epoch = epoch_i
            model_copy = copy.deepcopy(model)
            least_val_loss = avg_val_loss
            torch.save(model_copy, save_path)

        if use_wandb:
            wandb.log({"Epoch": epoch_i,
                        "Average training loss": avg_train_loss,
                        "Average validation loss":avg_val_loss,
                        "cur_least_epoch":cur_least_epoch})


def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-W', '--wandb', action=argparse.BooleanOptionalAction, help='For Wandb logging')
    parser.add_argument('-NTT', '--no_token_type', action=argparse.BooleanOptionalAction, help='If model does not have token type ids')
    parser.add_argument('-Attr', '--use_attr', action=argparse.BooleanOptionalAction, help='Use attribute')
    parser.add_argument('-ExIm', '--use_ex_im', action=argparse.BooleanOptionalAction, help='Use explicit implict tag')
    parser.add_argument("-PC", "--prefix_choice", type=int, default=1, help="Style of input construction to the model")
    parser.add_argument("-MN", "--model_name", type=str, default="bert-base-uncased", help="Variant of the Transformer model for finetuning")
    parser.add_argument("-SN", "--save_name", type=str, default="bert_org_10_0.1_0.1", help="Model name for saving")
    parser.add_argument("-ML", "--max_len", type=int, default=512, help="max length for tokenizers")
    parser.add_argument("-HD", "--hidden_dim", type=int, default=768, help="Hidden dimension of the model output")
    parser.add_argument("-B", "--batch_size", type=int, default=10, help="Batch size (should be equal to the number of questions generated)")
    # parser.add_argument("-ACG", "--accumulate_gradients", type=int, default=1, help="Gradient Accumulation")
    parser.add_argument("-alpha1", "--alpha1", type=float, default=0.1, help="Hyperparameter multiplied with the distribution of model loss for each question")
    parser.add_argument("-alpha2", "--alpha2", type=float, default=0.1, help="Hyperparameter multiplied with the distribution of the rouge scores")
    parser.add_argument('-TFN', '--train_file_name', type=str, default="train_nucleus_flan_t5_large_org_16_sa_0.95_1.00_10.csv", help="Training File name")
    parser.add_argument('-VFN', '--valid_file_name', type=str, default="valid_nucleus_flan_t5_large_org_16_sa_0.95_1.00_10.csv", help="Validation File name")
    parser.add_argument("-L", "--learning_rate", type=float, default=5e-5, help="Learning Rate for training the Transformer Model")
    parser.add_argument("-E", "--num_epochs", type=int, default=10, help="Total Number of Epochs")
    params = parser.parse_args()
    return params

def main():
    # parse arguments
    args = add_params()

    # check wandb
    if args.wandb:
        wandb.init(project="Quest_Gen_Challenge", entity="ml4ed", name=args.save_name)    

    # Read data 
    input_dir = './data/results_org/'
    train_df = pd.read_csv(os.path.join(input_dir, args.train_file_name))
    val_df = pd.read_csv(os.path.join(input_dir, args.valid_file_name))

    # NOTE: Check if rouge scores are pre-computed 
    if 'rougeL' not in list(train_df.columns):
        # Append rouge scores
        append_rouge_score(train_df)
        append_rouge_score(val_df)
        print('Appended Rouge Scores!')

        # Save rouge scores
        train_df.to_csv(os.path.join(input_dir, args.train_file_name), index=False)
        val_df.to_csv(os.path.join(input_dir, args.valid_file_name), index=False)

    # get input
    train_context_ans, train_question = get_transformer_input(train_df, args.prefix_choice, args.use_attr, args.use_ex_im)
    val_context_ans, val_question = get_transformer_input(val_df, args.prefix_choice, args.use_attr, args.use_ex_im)

    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_tokenized_dict = tokenize(tokenizer, train_context_ans, train_question, max_len=args.max_len)
    val_tokenized_dict = tokenize(tokenizer, val_context_ans, val_question, max_len=args.max_len)
    print('Tokenized the input!')
    print('Length of sequence encoded:', train_tokenized_dict['input_ids'].shape[1])

    # get data loader 
    batch_size = args.batch_size # should be equal to the number of questions
    train_dataloader = get_dataloader(batch_size, train_tokenized_dict, train_df['Rouge_L'], args.no_token_type)
    val_dataloader = get_dataloader(batch_size, val_tokenized_dict, val_df['Rouge_L'], args.no_token_type)
    
    # select device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)

    # Initialize model
    model = RougeKLModel(model_name=args.model_name, hid_dim=args.hidden_dim).to(device)    

    # optimizer and scheduler 
    optimizer, scheduler = get_optimizer_scheduler(model, args.learning_rate, len(train_dataloader), args.num_epochs)

    # finetune the model 
    save_path = os.path.join('./code/ranking_kl/Checkpoints_sep', args.save_name)


    train(args.wandb, args.no_token_type, args.num_epochs, model, device, train_dataloader, val_dataloader, optimizer, scheduler, args.alpha1, args.alpha2, save_path)

if __name__ == '__main__':
    main()