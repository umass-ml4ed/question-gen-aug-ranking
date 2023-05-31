import pandas as pd
import re
import string


def normalize(text):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def remove_duplicates(df, name):
    # Remove duplicates
    print(f"No of {name} samples before removing duplicates: {len(df)}")
    # Use generated_question_normalized and not generated_question_original to remove duplicates since the former is used for bleurt score
    df = df.drop_duplicates(subset=["pairID", "generated_question_normalized"])
    print(f"No of {name} samples after removing duplicates: {len(df)}")

    return df


def explode(df, col1, col2):
    df['tmp']=df.apply(lambda row: list(zip(row[col1],row[col2])), axis=1) 
    df=df.explode('tmp')
    df[[col1,col2]]=pd.DataFrame(df['tmp'].tolist(), index=df.index)
    df = df.drop(columns=['tmp'])
    
    return df


def rank(df_test, top_one=False):
    # Add predictions to df_test
    df_test = explode(df_test, "generated_question", "score")
    # Remove duplicates on pair id and normalized generated question
    df_test['generated_question_normalized'] = df_test['generated_question'].apply(normalize)
    df_test = remove_duplicates(df_test, "test")
    df_test = df_test.drop(columns=["generated_question_normalized"])

    if top_one:
        df_submission = df_test.groupby("pairID").apply(lambda x: x.sort_values("score", ascending=True).head(1))
    else:
        # Keep top 10 questions generated according to score predictions for each pair id
        # ascending=True since we want to keep the lowest perplexity scores
        df_submission = df_test.groupby("pairID").apply(lambda x: x.sort_values("score", ascending=True).head(10))
    print(f"No of test samples after ranking: {len(df_submission)}")

    return df_submission, df_test
