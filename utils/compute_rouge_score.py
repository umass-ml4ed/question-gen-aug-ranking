import os
import pandas as pd
import statistics
import argparse
from rouge_score import rouge_scorer
from code.utils.handle_data import load_df, RAW_DIR

def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-EFD', "--eval_folder", type=str, default="results_org", help="Folder containing evaluation file relative to data folder")
    parser.add_argument('-EFN', "--eval_filename", type=str, default="val.csv", help="Evaluation filename with timestamp and .csv extension")

    params = parser.parse_args()
    
    return params

def get_pairwise_preds(df_pred):
    attr, ex_or_im, loc_or_sum = [], [], []
    grp_wise_max_rouge, grp_wise_cor_pre, grp_wise_cor_rec = [], [], []
    grp_pair_ids = df_pred.groupby('pairID')
    for grp_full in grp_pair_ids:
        grp = grp_full[1]
        max_score = max(grp['rougeL_F1Score'])
        max_index = grp['rougeL_F1Score'].tolist().index(max_score)
        cor_pre, cor_rec = grp['rougeL_Precision'].iloc[max_index], grp['rougeL_Recall'].iloc[max_index]
        # print('max_score:', max_score)
        grp_wise_max_rouge.append(max_score)
        grp_wise_cor_pre.append(cor_pre)
        grp_wise_cor_rec.append(cor_rec)
        attr.append(grp['attribute'].tolist()[0])
        ex_or_im.append(grp['ex_or_im'].tolist()[0])
        loc_or_sum.append(grp['local_or_sum'].tolist()[0])
    
    reduced_df = pd.DataFrame()
    reduced_df['attribute'] = attr
    reduced_df['local_or_sum'] = loc_or_sum
    reduced_df['ex_or_im'] = ex_or_im
    reduced_df['rougeL_F1Score'] = grp_wise_max_rouge
    print('Mean ROUGE-L score:', statistics.mean(grp_wise_max_rouge))
    print("Mean ROUGE-L grouped by question attribute type:\n", reduced_df.groupby('attribute')['rougeL_F1Score'].agg(['mean', 'count']))
    print("Mean ROUGE-L grouped by question local vs summary:\n", reduced_df.groupby('local_or_sum')['rougeL_F1Score'].agg(['mean', 'count']))
    print("Mean ROUGE-L grouped by question explicit vs implicit:\n", reduced_df.groupby('ex_or_im')['rougeL_F1Score'].agg(['mean', 'count']))

    return statistics.mean(grp_wise_max_rouge), statistics.mean(grp_wise_cor_pre), statistics.mean(grp_wise_cor_rec)

def main():
    args = add_params()
    folder = os.path.join(RAW_DIR, args.eval_folder)
    pred_df = load_df(args.eval_filename, folder)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores_pre, scores_rec, scores_f1 = [], [], []
    for i, row in pred_df.iterrows():
        score = scorer.score(row['question'], row['generated_question'])
        scores_pre.append(score['rougeL'].precision)
        scores_rec.append(score['rougeL'].recall)
        scores_f1.append(score['rougeL'].fmeasure)
    
    pred_df['rougeL_Precision'] = scores_pre
    pred_df['rougeL_Recall'] = scores_rec
    pred_df['rougeL_F1Score'] = scores_f1
    
    mean_rl_f1, mean_rl_pre, mean_rl_rec = get_pairwise_preds(pred_df)
    
    print('\n#### Overall Mean Top-N Rouge-L Scores ####')
    print('Precision: {:.4f}'.format(mean_rl_pre))
    print('Recall: {:.4f}'.format(mean_rl_rec))
    print('F1 score: {:.4f}'.format(mean_rl_f1))

    
if __name__ == '__main__':
    main()