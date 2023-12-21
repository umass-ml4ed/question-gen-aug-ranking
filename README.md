# [Improving Reading Comprehension Question Generation with Data Augmentation and Overgenerate-and-rank](https://arxiv.org/abs/2306.08847)

In this repository we present the code to our paper "Improving Reading Comprehension Question Generation with Data Augmentation and Overgenerate-and-rank" by [Nischal Ashok Kumar](https://nish-19.github.io/), [Nigel Fernandez](https://www.linkedin.com/in/ni9elf/), [Zichao Wang](https://zw16.web.rice.edu/) and [Andrew Lan](https://people.umass.edu/~andrewlan/). We propose two methods, viz, Data Augmentation and Over-Generate-and-Rank that significantly improve the question generation performance on reading comprehension datasets. 

For any questions please [email](mailto:nashokkumar@umass.edu) or raise an issue.

If you find our code or paper useful, please consider citing:
```
@inproceedings{ashok-kumar-etal-2023-improving,
    title = "Improving Reading Comprehension Question Generation with Data Augmentation and Overgenerate-and-rank",
    author = "Ashok Kumar, Nischal  and
      Fernandez, Nigel  and
      Wang, Zichao  and
      Lan, Andrew",
    editor = {Kochmar, Ekaterina  and
      Burstein, Jill  and
      Horbach, Andrea  and
      Laarmann-Quante, Ronja  and
      Madnani, Nitin  and
      Tack, Ana{\"\i}s  and
      Yaneva, Victoria  and
      Yuan, Zheng  and
      Zesch, Torsten},
    booktitle = "Proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.bea-1.22",
    doi = "10.18653/v1/2023.bea-1.22",
    pages = "247--259",
    abstract = "Reading comprehension is a crucial skill in many aspects of education, including language learning, cognitive development, and fostering early literacy skills in children. Automated answer-aware reading comprehension question generation has significant potential to scale up learner support in educational activities. One key technical challenge in this setting is that there can be multiple questions, sometimes very different from each other, with the same answer; a trained question generation method may not necessarily know which question human educators would prefer. To address this challenge, we propose 1) a data augmentation method that enriches the training dataset with diverse questions given the same context and answer and 2) an overgenerate-and-rank method to select the best question from a pool of candidates. We evaluate our method on the FairytaleQA dataset, showing a 5{\%} absolute improvement in ROUGE-L over the best existing method. We also demonstrate the effectiveness of our method in generating harder, {``}implicit{''} questions, where the answers are not contained in the context as text spans.",
}
```


## Contents 

1. [Installation](#installation) 
2. [Finetuning](#finetuning)
3. [Inference/ Generation](#inferencegeneration)
4. [Data Augmentation](#data-augmentation) 
5. [Ranking](#ranking)
6. [ROUGE Score Computation](#rouge-scores-computation)

## Installation

A [Wandb](https://wandb.ai/site) account is required to log train-test-val information (loss, metrics, etc). Best model checkpoints are saved locally.


To install the necessary libraries for this codebase: 

```
conda env create -f environment.yml
```

## Finetuning

Use the ```-h``` option for displaying a list of all arguments and their descriptions. 

To finetune an encoder-decoder model (T5/BART):

```
python -m finetune.finetune_org \
    -W -MT T -MN google/flan-t5-large \
    -N flan_t5_large
```

The code accepts a list of arguments which are defined in the ```add_params``` function. 

The trained model checkpoint gets saved in the ```Checkpoints_org``` folder. 

## Inference/Generation

To get the inference/ generation using a pre-trained model: 

```
python -m finetune.inference_org \
    -MT T -MN google/flan-t5-large \
    -N flan_t5_large -DS N \
    -PS 0.9 -NS 10
```

The csv file containing the generations are saved in ```results_org```

## Data Augmentation

### Generate Synthetic Data

* **Genenerating Extra Data**

```
python -m prompt.get_aug \
    -SD -NK 6 -FN 1 \
    -N 4
```

Run the above code for fold numbers ```FN``` from 1 to 5. 

* **Clean Extra Data**

```
python -m prompt.clean_aug 

python -m prompt.filter_aug 
```

* **Generate and Evaluate Answers**

```
python -m prompt.get_answer 

python -m prompt.evaluate_answer 
```

* **Select Samples for Augmentation**

```
python -m prompt.sel_augment 
```

### Fine-tune on Augmented Dataset
```
python -m finetune.finetune_org_data_augment \
    -W -MT T -MN google/flan-t5-large \
    -N flan_t5_large_aug_0.8 -LAM 0.8
```

### Inference/Generation on Augmented Dataset 
```
python -m finetune.inference_org \
    -MT T -MN google/flan-t5-large \
    -N flan_t5_large -DS N \
    -PS 0.9 -NS 10
```

## Ranking

### Perplexity Ranking
```
python -m ranking_perplexity.generate \
    -MT T -MN google/flan-t5-large \
    -N flan_t5_large_aug_0.8 -DS N \
    -PS 0.9 -NS 10
```

### Distribution Ranking

First, run the [Inference/Generation](#inferencegeneration) step to get the 10 different generations per sample. 

* **Finetuning Distribution Ranking-Based Model**
```
python -m ranking_kl.bert_rank \
    -W -Attr -ExIm -MN YituTech/conv-bert-base \
    -SN convbert_org_10_0.001_0.01 \
    -alpha1 0.001 -alpha2 0.01
```

* **Predictions from Distribution Ranking-Based Model**
```
python -m ranking_kl.bert_rank_inference \
    -Attr -ExIm -MN YituTech/conv-bert-base \
    -SN convbert_org_10_0.001_0.01
```

The csv file containing the generations are saved in ```results_rank_kl```


## ROUGE Scores Computation

To compute the ROUGE Scores

```
python -m utils.compute_rouge_score \
    --eval_folder results_org \
    --eval_filename flan_t5_large_org.csv
```