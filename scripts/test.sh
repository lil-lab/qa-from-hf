# eval_batch_size -> batch size during evaluation
# n_best_size -> number of possible predictions to consider during evaluation
# max_answer_length -> maximum number of answer tokens
# initialize_model_from_checkpoint -> path to the folder containing checkpoint to be evaluated
# checkpoint_name -> name of the checkpoint to be evaluated (the actual path would be ${initialize_model_from_checkpoint}/${checkpoint_name})
# output_dir -> path to the folder where the evaluation results would be saved
# could specify checkpoints from round 1 to 9 in checkpoint_list and do evaluation at once

checkpoint_list=("checkpoint_path")
r=1

for i in "${checkpoint_list[@]}"; do   
    for checkpoint in $i
    do
      echo "$checkpoint | round $r"
      python train_bandit.py \
          --do_eval \
          --eval_test \
          --model microsoft/deberta-v3-base \
          --eval_batch_size 32  \
          --max_seq_length 512 \
          --doc_stride 512 \
          --eval_metric f1 \
          --na_prob_thresh 0 \
          --output_dir  ${checkpoint} \
          --test_file data/test_files.txt \
          --initialize_model_from_checkpoint ${checkpoint} \
          --version_2_with_negative \
          --checkpoint_name saved_checkpoint  \
          --n_best_size 20 \
          --max_answer_length 30 \
          --prepend_title \
          --round_index ${r} \
          --add_classifier
      r=$((r+1))
    done
done