# qa-from-hf
Code for [_Continually Improving Extractive QA via Human Feedback_](). Please contact the first authors by email if you have any question.

## Table of Contents
- [Basics](#basics)
- [Data](#data)
- [Installation](#installation)
- [Training Piepeline](#training-pipeline)
- [Evaluation](#evaluation)
- [Reproduction](#reproduction)
  - [Long-Term Study](#long-term-study)
  - [Analysis on Model Variants](#analysis-on-model-variants)
- [Citation](#citation)

## Basics
Brief intro to each folder and file at the root:
1. `data-collection/`: Examples and qualification tests designed for our user study. 
2. `data/`: All the data we collected for both the long-term study and analysis on model variants. You could use these data to reproduce our results. 
3. `scripts/`: Example scripts for training and testing the models.
4. `src/`: `data.py` is the script for loading the data; `eval.py` is the script for evaluation.
5. `src_analysis/`: Scripts for analyzing the results. 
6. `src_utils/`: Miscellaneous utility functions.
7. `generate_prob.py`: 
8. `random_indices_squad2.txt`:
9. `random_indices_tydi.txt`:
10. `model.py`: Script for model defination.
11. `rehearsal.py`: Training script.

To Do: double checking - we don't need the run_tydi.py?


## Data
You can find all the data in `data` folder:
- `train/`: Feedback data collected in the long-term deployment study.
- `train_parallel/`: Feedback data collected in the model variant study.
- `Dev.jsonl.gz`: The development set we use for hyperparameter tuning. We collected this set individually. 
- `static-test.jsonl.gz`: A static test sets we collected separately for validation during development.
- `full-test-long-term.jsonl.gz`: Full test set collected concurrently with the feedback data during the long-term study.
- `full-test-parallel.jsonl.gz`: Full test set collected concurrently with the feedback data during the study of different model variants. 
- `tydiqa-v1.0-dev.jsonl.gz`: TyDiQA development set. We only consider the English portion and exclude the Yes/No questions. 
- `test_feedback.txt`: This text file should contain the dataset you would like to evaluate your model on. Each line is formatted as \[feedback type\]\\t\[file name\].
- ToDo: add the 512-SQuAD2 example files
- ToDo: link to the NewsQA training data



## Installation
1. This project is developed in Python 3.6. Using Conda to set up a virtual environment is recommended.

2. Install the required dependencies. 
    ```
    pip install -r requirements.txt
    ```
    
3. Install PyTorch from http://pytorch.org/.


## Training Pipeline
### 1. Initial Training
We train an initial DeBERTaV3 model on a set of random sampled 512 SQuAD2 examples, or on NewsQA.
- SQuAD2-initialized model: Run `pretrain.sh` after replacing `output_dir` with the directory you want to save the model.
- NewsQA-initialized model: Run `pretrain.sh` after changing `data_type` to `newsqa`, removing `--num_initial_data 512`, and replacing `output_dir` with the directory you want to save the model.

### 2. Bandit Learning
We iteratively improve the model via multiple rounds of user interaction.

ToDo: add eaxamples? hyperparamter part


## Evaluation
ToDo: add instruction on how to evaluate the model

## Reproduction

### Long-Term Study
1. Follow the steps in [Initial Training](#initial-training) to get a 512-SQuAD2 initial model.
2. To be completed

### Analysis on Model Variants

To be completed




## Citation
```

```
