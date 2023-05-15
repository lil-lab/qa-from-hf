#!/bin/bash
# Readme:
# data_file -> path to the collected data
# outfile -> path to the output file (will write to that file)
# checkpoint -> model checkpoint for collecting that round
# model -> model type
r_index=10
model_source="fewer"
outfile="data/bandit_parallel/train/round${r_index}/train-data-parallel-round${r_index}-200-wprob-${model_source}.jsonl"
c_name="2023042511370472"
checkpoint="parallel_exp/round10/fewer-r10-20230426123730412874/saved_checkpoint"

python generate_prob.py --data_file data/collected_data/train/parallel/round${r_index}/round${r_index}-${c_name}-${model_source}-feedback.jsonl.gz \
                             --outfile ${outfile} \
                             --model microsoft/deberta-v3-base \
                             --checkpoint ${checkpoint} \
                             --add_classifier

gzip ${outfile}