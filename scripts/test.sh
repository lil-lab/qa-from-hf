checkpoint_list=("checkpoint_path")
r=1

for i in "${checkpoint_list[@]}"; do   
    for checkpoint in $i
    do
      echo "$checkpoint | round $r"
      python rehearsal.py \
          --do_eval \
          --eval_test \
          --model microsoft/deberta-v3-base \
          --eval_batch_size 32  \
          --max_seq_length 512 \
          --doc_stride 512 \
          --eval_metric f1 \
          --na_prob_thresh 0 \
          --output_dir  ${checkpoint} \
          --test_file data/test_feedback.txt \
          --initialize_model_from_checkpoint ${checkpoint} \
          --version_2_with_negative \
          --test_data_type - \
          --checkpoint_name saved_checkpoint  \
          --n_best_size 20 \
          --max_answer_length 30 \
          --prepend_title \
          --round_index ${r} \
          --add_classifier
      r=$((r+1))
    done
done