# Improving Reading Comprehension Question Generation with Data Augmentation and Overgenerate-and-rank

In this repository we present the code to our paper "Improving Reading Comprehension Question Generation with Data Augmentation and Overgenerate-and-rank" by [Nischal Ashok Kumar](https://nish-19.github.io/), [Nigel Fernandez](https://www.linkedin.com/in/ni9elf/), [Zichao Wang](https://zw16.web.rice.edu/) and [Andrew Lan](https://people.umass.edu/~andrewlan/). We propose two methods, viz, Data Augmentation and Over-Generate-and-Rank that significantly improve the question generation performance on reading comprehension datasets. 

For any questions please [email](mailto:nashokkumar@umass.edu) or raise an issue.

If you find our code or paper useful, please consider citing:

<TODO: Add citation>

## Contents 

1. [Installation](#installation) 
2. [Finetuning](#finetuning)
3. [Inference/ Generation](#inferencegeneration)
4. [Data Augmentation](#data-augmentation) 
5. [Ranking](#ranking)
6. [ROUGE Score Computation](#rouge-scores-computation)

## Installation

To install the libraries need for running this code: 

```
conda env create -f environment.yml
```

## Finetuning

To finetune an encoder-decoder model (T5/BART):

```
python -m code.finetune.finetune_org \
    -W -MT T -MN google/flan-t5-large \
    -N flan_t5_large
```

The code accepts a list of arguments which are defined in the ```add_params``` function. 

The trained model checkpoint gets saved in the ```Checkpoints_org``` folder. 

## Inference/Generation

To get the inference/ generation using a pre-trained model: 

```
python -m code.finetune.inference_org \
    -MT T -MN google/flan-t5-large \
    -N flan_t5_large -DS N \
    -PS 0.9 -NS 10
```

The csv file containing the generations are saved in ```results_org```

## Data Augmentation

### Generate Synthetic Data

* **Genenerating Extra Data**

```
python -m code.prompt.get_aug \
    -SD -NK 6 -FN 1 \
    -N 4
```

Run the above code for fold numbers ```FN``` from 1 to 5. 

* **Clean Extra Data**

```
python -m code.prompt.clean_aug 

python -m code.prompt.filter_aug 
```

* **Generate and Evaluate Answers**

```
python -m code.prompt.get_answer 

python -m code.prompt.evaluate_answer 
```

* **Select Samples for Augmentation**

```
python -m code.prompt.sel_augment 
```

### Fine-tune on Augmented Dataset
```
python -m code.finetune.finetune_org_data_augment \
    -W -MT T -MN google/flan-t5-large \
    -N flan_t5_large_aug_0.8 -LAM 0.8
```

### Inference/Generation on Augmented Dataset 
```
python -m code.finetune.inference_org \
    -MT T -MN google/flan-t5-large \
    -N flan_t5_large -DS N \
    -PS 0.9 -NS 10
```

## Ranking

### Perplexity Ranking
```
python -m code.ranking_perplexity.generate \
    -MT T -MN google/flan-t5-large \
    -N flan_t5_large_aug_0.8 -DS N \
    -PS 0.9 -NS 10
```

### Distribution Ranking

* **Finetuning Distribution Ranking-Based Model**
```
python -m code.ranking_kl.bert_rank \
    -W -Attr -ExIm -MN YituTech/conv-bert-base \
    -SN convbert_org_10_0.001_0.01 \
    -alpha1 0.001 -alpha2 0.01
```

* **Predictions from Distribution Ranking-Based Model**
```
python -m code.ranking_kl.bert_rank_inference \
    -Attr -ExIm -MN YituTech/conv-bert-base \
    -SN convbert_org_10_0.001_0.01
```

The csv file containing the generations are saved in ```results_rank_kl```


## ROUGE Scores Computation

To compute the ROUGE Scores

```
python -m code.utils.compute_rouge_score \
    --eval_folder results_org \
    --eval_filename flan_t5_large_org.csv
```