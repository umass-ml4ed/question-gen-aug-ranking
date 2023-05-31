import os 
import pandas as pd
from string import punctuation

def clean_response(response):
    question = str(response).split('?')[0]
    question = question.replace('<Question>', '') # remove tag if exists
    clean_question = question.strip(punctuation) + '?' # remove all leading and trailing puntuations 
    return clean_question

def main():
    output_path = 'output'
    clean_path = 'clean_aug'
    for i in range(5):
        print('Augment File: {:d}'.format(i+1))
        file_name = 'augment_fold_{:d}.csv'.format(i+1)
        augment_df = pd.read_csv(os.path.join(output_path, file_name)) 
        augment_df['Response_1'] = augment_df['Response_1'].apply(clean_response)
        augment_df['Response_2'] = augment_df['Response_2'].apply(clean_response)
        augment_df.to_csv(os.path.join(clean_path, file_name), index=False)

if __name__ == '__main__':
    main()