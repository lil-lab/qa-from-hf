# To Be Completed...
# qa-from-hf
Code for [_Continually Improving Extractive QA via Human Feedback_](). Please contact the first authors by email if you have any question.

## Table of Contents
- [qa-from-hf](#qa-from-hf)
  - [Table of Contents](#table-of-contents)
  - [Basics](#basics)
  - [Data](#data)
  - [Installation](#installation)
  - [Reproduction](#reproduction)
    - [Initial Training](#initial-training)
    - [Bandit Learning](#bandit-learning)
  - [Evaluation](#evaluation)
  - [Citation](#citation)

## Basics
Brief intro to each folder and file at the root:
1. `data-collection/`: Examples and qualification tests designed for our user study. 
2. `data/`: All the data we collected for both the long-term study and analysis on model variants. You could use these data to reproduce our results. 
3. `scripts/`: Example scripts for training and testing the models.
4. `src/`: `data.py` is the script for loading the data; `eval.py` is the script for evaluation.
5. `src_analysis/`: Scripts for analyzing the results. 
6. `src_utils/`: Miscellaneous utility functions.
7. `generate_prob.py`: The script we used to store the generation probability in the data files.
8. `random_indices_squad2.txt`: The random indices we use to shuffle the SQuAD2.0 initial data. Will need this file to reproduce the initial model.
9.  `model.py`: Script for model defination.
10. `rehearsal.py`: Training script for bandit learning.
11. `run_tydi.py`: Training script for initial model training.


## Data
We are using [squad_v2](https://huggingface.co/datasets/squad_v2) on huggingface for SQuAD2-initialized models.   
We use the NewsQA data from [TODO: this_link](https://newsqa_link.link). 

You can find all the other data in `data` folder:
- `train/`: Feedback data collected in the long-term deployment study.
- `train_parallel/`: Feedback data collected in the model variant study.
- `Dev.jsonl.gz`: The development set we use for hyperparameter tuning. We collected this set individually. 
- `static-test.jsonl.gz`: A static test sets we collected separately for validation during development.
- `full-test-long-term.jsonl.gz`: Full test set collected concurrently with the feedback data during the long-term study.
- `full-test-parallel.jsonl.gz`: Full test set collected concurrently with the feedback data during the study of different model variants. 
- `tydiqa-v1.0-dev.jsonl.gz`: TyDiQA development set. We only consider the English portion and exclude the Yes/No questions. 
- `test_feedback.txt`: This text file should contain the dataset you would like to evaluate your model on. Each line is formatted as \[feedback type\]\\t\[file name\].
- ToDo: add the 512-SQuAD2 example files -> we don't need this since we are using [squad_v2](https://huggingface.co/datasets/squad_v2) on huggingface for SQuAD2-initialized models. And we use `random_indices_squad2.txt` to shuffle the dataset. 
- ToDo: link to the NewsQA training data



## Installation
1. This project is developed in Python 3.6. Using Conda to set up a virtual environment is recommended.

2. Install the required dependencies. 
    ```
    pip install -r requirements.txt
    ```
    
3. Install PyTorch from http://pytorch.org/.


## Reproduction
### Initial Training
We train an initial DeBERTaV3 model on a set of random sampled 512 SQuAD2 examples, or on NewsQA.
- 512-SQuAD2-initialized model: Run `pretrain.sh` after replacing `output_dir` with the directory you want to save the model.
- 128-SQuAD2-initialized model: Run `pretrain.sh` after changing `num_initial_data` to `128`, and replacing `output_dir` with the directory you want to save the model.
- NewsQA-initialized model: Run `pretrain.sh` after changing `data_type` to `newsqa`, removing `--num_initial_data 512`, and replacing `output_dir` with the directory you want to save the model.
- To Do: add the model without CLS head

### Bandit Learning
We iteratively improve the model via multiple rounds of user interaction. At each round, the pipelien is to specify the feedback data for training, and then conduct the bandit learning. Concrete steps are as follows:

1. Specifiy Training Data: Before each round of bandit learning, you should specify the training data by modifying `train_files.txt`. To do so, you could simply run `src_utils/write_data_file.py` with corresponding arguments.  

An example script for long-term experiments:  

      python src_utils/write_data_file.py --exp long-term --r_idx 1

An example script for experiments on different model variants:  

      python src_utils/write_data_file.py --exp variants  --r_idx 1 --variant fewer

`fewer` for fewer examples per round, to do: complete this part

2. Training: Run `rehearsal.py` to do bandit learning. We perform hyperparameter tuning on `num_train_epochs`, `learning_rate` and `entropy_coeff` as mentioned in the paper.   
An example script is provided below: (refer to `scripts/rehearsal.sh` for more details.)    
You should specify `output_dir` which is the the output directory (for storing the model and training log) and, `initialize_model_from_checkpoint` which is the path to the model that you want to start with. For Round 1, this model path should be that of an initial model obtrained from inital training.

        python rehearsal.py   --do_train  \
                              --do_eval   \
                              --train_file train_files.txt   \
                              --output_dir [output_dir]   \
                              --initialize_model_from_checkpoint [model_path]   \
                              --dev_file data/Dev-400.jsonl.gz   \
                              --num_train_epochs 30   \
                              --learning_rate 3e-6   \
                              --entropy_coeff 5.0   \
                              --train_batch_size 35   \
                              --version_2_with_negative   \
                              --prepend_title   \
                              --load_log_prob   \
                              --tag example \
                              --round_index 1   \
                              --turn_off_dropout   \
                              --add_classifier   \
                              --rehearsal   


For the next round of bandit learning, repeat the above 2 steps. At every round, remember to change `\[model_path\]` in step 2 to be the best-performing model on the development set from the previous round.  

## Evaluation
ToDo: add instruction on how to evaluate the model


To be completed




## Citation
```

```
