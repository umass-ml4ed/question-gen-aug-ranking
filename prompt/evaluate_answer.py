import os
import argparse 
import pandas as pd 
import string 
import re 
from rouge_score import rouge_scorer

def normalize(text):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_answer_tag(text):
        return text.replace('<answer>', '')

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(remove_answer_tag(lower(str(text))))))

# def normalize(text):
#     text = str(text).strip('<Answer>').lower().strip(string.punctuation)
#     return text

def clean_answer(df, args):
    df['answer'] = df['answer'].apply(normalize)
    df['Org Answer'] = df['Org Answer'].apply(normalize)
    for j in range(args.num_responses):
        df['R{:d} Answer'.format(j+1)] = df['R{:d} Answer'.format(j+1)].apply(normalize)
    return df

def compute_rouge_score(col1, col2, args):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    all_scores = []
    for ans1, ans2 in zip(col1, col2):
        score = scorer.score(str(ans1), str(ans2))
        all_scores.append(score['rouge1'].fmeasure)
    return all_scores

def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-SD', '--selective_augmentation', action=argparse.BooleanOptionalAction, help='Augment all attribute except action and causal')
    parser.add_argument("-NR", "--num_responses", type=int, default=2, help="Number of generated questions for whom answers need to be generated")

    params = parser.parse_args()
    return params


def main():
    args = add_params()
    ans_dir = 'answer'
    output_dir = 'rouge'
    if args.selective_augmentation:
        ans_dir = os.path.join(ans_dir, 'sel_aug')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_dir = os.path.join(output_dir, 'sel_aug')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i in range(5):
        filename = 'augment_fold_{:d}.csv'.format(i+1)
        print(filename)
        aug_df = pd.read_csv(os.path.join(ans_dir, filename))
        clean_aug_df = clean_answer(aug_df, args)
        for j in range(args.num_responses):
            rj_ans_score = compute_rouge_score(clean_aug_df['answer'], clean_aug_df['R{:d} Answer'.format(j+1)], args)
            rj_org_score = compute_rouge_score(clean_aug_df['Org Answer'], clean_aug_df['R{:d} Answer'.format(j+1)], args)
            clean_aug_df['r{:d}_ans_score'.format(j+1)] = rj_ans_score
            clean_aug_df['r{:d}_org_score'.format(j+1)] = rj_org_score
        output_path = os.path.join(output_dir, filename)
        clean_aug_df.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()