import os 
import pandas as pd 
import argparse

def add_row(row, j, story, content, ans, ques, ls, attr, ei):
    story.append(row['story_name'])
    content.append(row['content'])
    ans.append(row['R{:d} Answer'.format(j+1)])
    ques.append(row['Response_{:d}'.format(j+1)])
    ls.append(row['local_or_sum'])
    attr.append(row['attribute'])
    ei.append(row['ex_or_im'])

def choose_row(row, j, args, dist_match, story, content, ans, ques, ls, attr, ei):
    if row['r{:d}_ans_score'.format(j+1)] >= 0.5 or row['r{:d}_org_score'.format(j+1)] >= 0.5:
        if args.selective_augmentation:
            if row['Org Answer'] != 'skip':
                add_row(row, j, story, content, ans, ques, ls, attr, ei)
        else:
            if row['attribute'] == 'prediction':
                prefix = ' '.join(row['Response_{:d}'.format(j+1)].split(' ')[:2])
            else:
                prefix = row['Response_{:d}'.format(j+1)].split(' ')[0]
            if prefix in dist_match[row['attribute']]:
                add_row(row, j, story, content, ans, ques, ls, attr, ei)


def filter_aug(aug_df, args):
    dist_match = {'action': ['what'], 'causal relationship': ['why'], 
                'character':['who'], 'feeling':['how'], 
                'outcome resolution': ['what'], 'prediction':['what will', 'how will'],
                'setting': ['where']}
    story, content, ans, ques, ls, attr, ei = [], [], [], [], [], [], []
    for i, row in aug_df.iterrows():
        # NOTE: Iterate over all responses 
        for j in range(args.num_responses):
            choose_row(row, j, args, dist_match, story, content, ans, ques, ls, attr, ei)
    return story, content, ans, ques, ls, attr, ei

def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-SD', '--selective_augmentation', action=argparse.BooleanOptionalAction, help='Augment all attribute except action and causal')
    parser.add_argument("-NR", "--num_responses", type=int, default=2, help="Number of generated questions for which answers need to be generated")

    params = parser.parse_args()
    return params

def main():
    args = add_params()
    rouge_dir = 'rouge'
    if args.selective_augmentation:
        rouge_dir = os.path.join(rouge_dir, 'sel_aug')
    all_story, all_content, all_ans, all_ques, all_ls, all_attr, all_ei = [], [], [], [], [], [], []
    for i in range(5):
        aug_filename = 'augment_fold_{:d}.csv'.format(i+1)
        print(aug_filename)
        aug_file_path = os.path.join(rouge_dir, aug_filename) 
        aug_df = pd.read_csv(aug_file_path)
        story, content, ans, ques, ls, attr, ei = filter_aug(aug_df, args)
        all_story.extend(story)
        all_content.extend(content)
        all_ans.extend(ans)
        all_ques.extend(ques)
        all_ls.extend(ls)
        all_attr.extend(attr)
        all_ei.extend(ei)
    df = pd.DataFrame()
    df['story_name'] = all_story
    df['content'] = all_content
    df['answer'] = all_ans
    df['question'] = all_ques
    df['local_or_sum'] = all_ls 
    df['attribute'] = all_attr 
    df['ex_or_im'] = all_ei
    # save df
    output_dir = 'filter'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if args.selective_augmentation:
        output_dir = os.path.join(output_dir, 'sel_aug')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    save_name = 'filter_aug.csv'
    save_path = os.path.join(output_dir, save_name)
    df.to_csv(save_path, index=False)
        
if __name__ == '__main__':
    main()