# qa-from-hf
Code for [_Continually Improving Extractive QA via Human Feedback_](https://arxiv.org/abs/2305.12473). Please contact the first authors by email if you have any question.

## Table of Contents
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
8. `random_indices_squad2.txt`: The random indices we use to shuffle the SQuAD2.0 initial data. Will need this file to reproduce our initial model.
9.  `model.py`: Script for model defination.
10. `train_bandit.py`: Training script for bandit learning.
11. `train_initial.py`: Training script for initial model training.


## Data
We are using [squad_v2](https://huggingface.co/datasets/squad_v2) dataset on Hugging Face for SQuAD2-initialized models.   
We use the NewsQA data from [TODO: this_link](https://newsqa_link.link). 

You can find all other data used in our paper in the `data` folder:
- `train/`: Feedback data collected in the long-term deployment study.
- `train_parallel/`: Feedback data collected in the model variant study.
- `Dev.jsonl.gz`: The development set we use for hyperparameter tuning. We collected this set individually. 
- `static-test.jsonl.gz`: A static test sets we collected separately for validation during development.
- `full-test-long-term.jsonl.gz`: Full test set collected concurrently with the feedback data during the long-term study.
- `full-test-parallel.jsonl.gz`: Full test set collected concurrently with the feedback data during the study of different model variants. 
- `tydiqa-v1.0-dev.jsonl.gz`: TyDiQA development set. We only consider the English portion and exclude the Yes/No questions. 
- `test_files.txt`: This text file should contain the dataset you would like to evaluate your model on. Each line is formatted as \[feedback type\]\\t\[file name\].




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


### Bandit Learning
We iteratively improve the model via multiple rounds of user interaction. At each round, the pipeline is to specify the feedback data for training, and then conduct the bandit learning. Concrete steps are as follows:

1. Specifiy Training Data: Before each round of bandit learning, you should specify the training data by modifying `train_files.txt`. To do so, you could simply run `src_utils/write_data_file.py` with corresponding arguments.  

An example script for long-term experiments:  

      python src_utils/write_data_file.py --exp long-term --r_idx 1

An example script for experiments on different model variants:  

      python src_utils/write_data_file.py --exp variants  --r_idx 1 --variant fewer

`fewer` for fewer examples per round,  `default` for default setup, `newsqa` for domain adaptation from NewsQA, `noclass` for ablation on classification head, and `weaker` for starting with a weaker initial model.

2. Training: Run `train_bandit.py` to do bandit learning. We perform hyperparameter tuning on `num_train_epochs`, `learning_rate` and `entropy_coeff` as mentioned in the paper.   
An example script is provided below: 
(refer to `scripts/train_bandit.sh` for more details)    

        python train_bandit.py   --do_train  \
                              --do_eval   \
                              --train_file train_files.txt   \
                              --output_dir [your_output_dir]   \
                              --initialize_model_from_checkpoint [your_model_path]   \
                              --checkpoint_name [your_model_name]  \
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
You should specify `output_dir` which is the the output directory (for storing the model and training log) and, `initialize_model_from_checkpoint` and `checkpoint_name` which are the path to and name of the model that you want to start with. For Round 1, this model path should be that of an initial model obtrained from inital training. For ablation on classification head, remeber to remove `--add_classifier`.


For the next round of bandit learning, repeat the above 2 steps. At every round, remember to change `initialize_model_from_checkpoint` in step 2 to be the best-performing model on the development set from the previous round.  

## Evaluation
First, you need to specify which datasets/files you would like to evaluate your model on. You can modify `test_files.txt` to indicate which files you would like to test on. Each line represents a test file, and should be formatted as \[feedback type\]\\t\[file name\].  The example `test_files.txt` in the repo lists all possible datasets that you can evaluate on. 

To conduct the evaluation, run `train_bandit.py` with proper arguments: `output_dir` which is the the output directory for evaluation results, `initialize_model_from_checkpoint` and `checkpoint_name` which are the path to and name of the model that you want to evaluate.
An example script is as follows: (refer to `scripts/test.sh` for more details)  


      python train_bandit.py \
            --do_eval \
            --eval_test \
            --model microsoft/deberta-v3-base \
            --test_file data/test_files.txt \
            --output_dir  [your_output_dir] \
            --initialize_model_from_checkpoint [your_model_path]  \
            --checkpoint_name [your_model_name]  \
            --version_2_with_negative \
            --prepend_title \
            --add_classifier          


The results of the evaluation will be stored at the specified `output_dir` and printed as standard output.


## Citation
      @InProceedings{Gao23continually,
      author    = {Ge Gao, Hung-Ting Chen, Yoav Artzi, and Eunsol Choi},
      title     = {Continually Improving Extractive QA via Human Feedback},
      booktitle = {EMNLP},
      year      = {2023}
      }    
