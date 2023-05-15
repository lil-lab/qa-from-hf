#!/bin/bash
# replace [output_dir] with the directory you want to save the model
# replace data_type with newsqa and remove "--num_initial_data 512" to train on NewsQA dataset
python run_tydi.py \
  --do_train \
  --do_eval \
  --model microsoft/deberta-v3-base \
  --train_file data/initial_data/NewsQA-train.jsonl.gz \
  --dev_file data/initial_data/NewsQA-dev.jsonl.gz \
  --train_batch_size 10 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 5  \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --eval_per_epoch 5 \
  --max_seq_length 512 \
  --doc_stride 512 \
  --eval_metric f1 \
  --na_prob_thresh 0 \
  --version_2_with_negative \
  --prepend_title \
  --data_type squad2 \
  --output_dir [output_dir] \
  --num_initial_data 512 \
  --add_classifier \
  --random_indices random_indices_squad2.txt 