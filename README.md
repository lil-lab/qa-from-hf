# qa-from-hf
Code for [_Learning to Answer Questions from Human Feedback_]().

## Table of Contents
- [qa-from-hf](#qa-from-hf)
  - [Table of Contents](#table-of-contents)
  - [Basics](#basics)
  - [Data](#data)
  - [Installation](#installation)
  - [Reproduction](#reproduction)
    - [Data Prerequisite](#data-prerequisite)
      - [Pretraining](#pretraining)
      - [Bandit Learning](#bandit-learning)
    - [Pretraining](#pretraining-1)
    - [Long-Term Study](#long-term-study)
    - [Analysis on Model Variants](#analysis-on-model-variants)
  - [Citation](#citation)

## Basics
Brief intro for each folder:
1. `data/`: All the data we collected for both the long-term study and analysis on model variants. You could use these data to reproduce our results. 
2. `data-collection/`: Screenshots of examples and qualification tests for both our annotation tasks. 
3. `scripts/`: Example scripts for training and testing the models.
4. `src/`: `data.py` is the script for loading the data; `eval.py` is the script for evaluation.
5. `src_analysis/`: Scripts for analyzing the results. 
6. `src_utils/`: Miscellaneous utility functions.

## Data
You can find all the data in `data` folder, with the data collected from long-term study in `train` and the data collected from analysis on different model variants in `train_parallel`.   
The other files:
- `Dev.jsonl.gz`: The development set we use for hyperparameter tuning. We collected this set individually. 
- `full-test-long-term.jsonl.gz`: The test set we collected concurrently with the feedback during the long-term study.
- `full-test-parallel.jsonl.gz`: The test set we collected concurrently with the feedback during the study of different model variants. 
- `static-test.jsonl.gz`: The static test sets we collected separately. Just for understanding the performances of the models.
- `tydiqa-v1.0-dev.jsonl.gz`: TyDiQA development set. We consider only the English portion and exclude the Yes/No questions. 
- `test_feedback.txt`: The text file containing the file you would like to evaluate on. Each line is formatted as \[feedback type\]\\t\[file name\].

To obtain the data you need to reproduce the results, please refer to [Reproduction](#reproduction).

## Installation
1. This project is developed in Python 3.6. Using Conda to set up a virtual environment is recommended.

2. Install the required dependencies. 
    ```
    pip install -r requirements.txt
    ```
    
3. Install PyTorch from http://pytorch.org/.


## Reproduction
### Data Prerequisite
#### Pretraining
We are using [squad_v2](https://huggingface.co/datasets/squad_v2) on huggingface for SQuAD2-initialized models. 
For NewsQA, ...

#### Bandit Learning


### Pretraining 
1. SQuAD2-initialized model: Replace [output_dir] in `pretrain.sh` with the directory you want to save the model and run it.
2. NewsQA-initialized model: Do 1., replace `data_type` with `newsqa` and remove `--num_initial_data 512` in `pretrain.sh`. Then run `pretrain.sh`.

### Long-Term Study
1. Follow the steps in [Data Prerequisite](#DataPrerequisite).
2. 

### Analysis on Model Variants






## Citation
```

```
