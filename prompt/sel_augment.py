import os 
import random
import sys
import json 
import pandas as pd 
from collections import Counter
import argparse

def display_attr_stat(train_df, aug_df, to_add=False):
    train_attr_stats = dict(sorted(dict(Counter(train_df['attribute'])).items(), key= lambda item : item[1], reverse=True))
    aug_attr_stats = dict(sorted(dict(Counter(aug_df['attribute'])).items(), key= lambda item : item[1], reverse=True))
    attr_names = list(train_attr_stats.keys())
    train_attr_values = list(train_attr_stats.values())
    aug_attr_values = []
    for attr_name in attr_names:
        try:
            aug_attr_values.append(aug_attr_stats[attr_name])
        except KeyError:
            aug_attr_values.append(0)
    attr_df = pd.DataFrame()
    attr_df['Attr Name'] = attr_names
    attr_df['Train Count'] = train_attr_values
    attr_df['Aug Count'] = aug_attr_values
    if to_add:
        total_values = [tv + av for tv, av in zip(train_attr_values, aug_attr_values)]
        attr_df['Total Count'] = total_values
    print(attr_df)
    return attr_df

def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-SD', '--selective_augmentation', action=argparse.BooleanOptionalAction, help='Augment all attribute except action and causal')
    parser.add_argument('-RB', '--remove_balance', action=argparse.BooleanOptionalAction, help='Remove extra action and causal attributes from the actual training data')
    params = parser.parse_args()
    return params

def main():
    # argparse 
    args = add_params()
    random.seed(37)

    # NOTE: Aug df
    filter_dir = 'filter'
    if args.selective_augmentation:
        filter_dir = os.path.join(filter_dir, 'sel_aug')
    filename = 'filter_aug.csv'
    aug_df = pd.read_csv(os.path.join(filter_dir, filename)).sample(frac=1, random_state=37)

    # NOTE: Train data
    data_path = '../data/train.json'
    train_data = []
    with open(data_path, 'r') as infile:
        for line in infile:
            train_data.append(json.loads(line))
    # shuffle train data
    random.shuffle(train_data)
    train_df = pd.DataFrame(train_data)

    # Display stats 
    print('Attribute Distribution')
    attr_df = display_attr_stat(train_df, aug_df, to_add=True)

    # NOTE: Type 0 (Dump only the augment data)
    org_data = []
    aug_df_cols = list(aug_df.columns)
    for i, row in aug_df.iterrows():
        vals = row.values
        aug_dict = {col:val for col, val in zip(aug_df_cols, vals)}
        # augment to the train_data
        org_data.append(aug_dict)

    # Store the augmented data
    store_dir = '../data'
    store_filename = 'prompt_only_data.json'
    with open(os.path.join(store_dir, store_filename), 'w') as f:
        for d in org_data:
            json.dump(d, f)
            f.write('\n')

    # # NOTE: Aug Type 1: Complete Data Augmentation 
    # org_data = train_data.copy()
    # aug_df_cols = list(aug_df.columns)
    # for i, row in aug_df.iterrows():
    #     vals = row.values
    #     aug_dict = {col:val for col, val in zip(aug_df_cols, vals)}
    #     # augment to the train_data
    #     org_data.append(aug_dict)
    
    # # Store the augmented data
    # store_dir = '../data/FairytaleQA'
    # store_filename = 'prompt_aug_full_train.json'
    # with open(os.path.join(store_dir, store_filename), 'w') as f:
    #     for d in org_data:
    #         json.dump(d, f)
    #         f.write('\n')

    # # NOTE: Aug Type 2: Augment only character, feeling, outcome resolution, setting and prediction
    # org_data = train_data.copy()
    # aug_df_cols = list(aug_df.columns)
    # aug_choices = ['character', 'feeling', 'outcome resolution', 'setting', 'prediction']
    # for i, row in aug_df.iterrows():
    #     if row['attribute'] in aug_choices:
    #         vals = row.values
    #         aug_dict = {col:val for col, val in zip(aug_df_cols, vals)}
    #         # augment to the train_data
    #         org_data.append(aug_dict)
    
    # # Store the augmented data
    # store_dir = '../data/FairytaleQA'
    # store_filename = 'prompt_aug_selective_train.json'
    # with open(os.path.join(store_dir, store_filename), 'w') as f:
    #     for d in org_data:
    #         json.dump(d, f)
    #         f.write('\n')

    # NOTE: Aug Type 3: Augment based on count (upto least total count)
    least_count = min(attr_df['Total Count'])
    allow_aug = {}
    for i, row in attr_df.iterrows():
        allow_aug[row['Attr Name']] = least_count - row['Train Count']

    # Start Augmentation
    org_data = train_data.copy()
    aug_df_cols = list(aug_df.columns)
    for i, row in aug_df.iterrows():
        if allow_aug[row['attribute']] > 0:
            vals = row.values
            aug_dict = {col:val for col, val in zip(aug_df_cols, vals)}
            # augment to the train_data
            org_data.append(aug_dict)
            allow_aug[row['attribute']] -= 1
    
    remove_instances = {'action':attr_df['Train Count'][attr_df['Attr Name'] == 'action'].values[0] - least_count, 'causal relationship':attr_df['Train Count'][attr_df['Attr Name'] == 'causal relationship'].values[0] - least_count}
    if args.remove_balance:
        for data in org_data:
            if data['attribute'] == 'action' or data['attribute'] == 'causal relationship':
                if remove_instances[data['attribute']] > 0:
                    org_data.remove(data)
                    remove_instances[data['attribute']] -= 1
        suffix = 'rb'
    else:
        suffix = ''

    # Store the augmented data
    store_dir = '../data'
    store_filename = 'prompt_aug_control_count_{:s}.json'.format(suffix)
    with open(os.path.join(store_dir, store_filename), 'w') as f:
        for d in org_data:
            json.dump(d, f)
            f.write('\n')
    
    # Verify attribute distribution
    print("After Augmentation")
    attr_df = display_attr_stat(train_df, pd.DataFrame(org_data), to_add=False)




if __name__ == '__main__':
    main()

