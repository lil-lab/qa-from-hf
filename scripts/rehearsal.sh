# hyperparameters - num_train_epochs, learning_rate, entropy_coeff
# remove --wandb to disable wandb logging
# remove --add_classifier to disable classifier for answerability

# train_files.txt should contain 2 lines, one being the path to the current round data and the other to the data from all previous rounds
# replace output_dir with the directory to the saved model (and log)
# replace model_path with the path to the model from previous round (or initial model)
python rehearsal.py   --do_train  \
                      --do_eval   \
                      --model microsoft/deberta-v3-base   \
                      --train_file train_files.txt   \
                      --output_dir [output_dir]   \
                      --initialize_model_from_checkpoint [model_path]   \
                      --dev_file data/Dev-400.jsonl.gz   \
                      --num_train_epochs 30   \
                      --learning_rate 3e-6   \
                      --entropy_coeff 5.0   \
                      --train_batch_size 35   \
                      --eval_batch_size 6    \
                      --eval_per_epoch 4   \
                      --version_2_with_negative   \
                      --prepend_title   \
                      --eval_metric f1   \
                      --tag fewer2   \
                      --load_log_prob   \
                      --wandb   \
                      --round_index 1   \
                      --turn_off_dropout   \
                      --add_classifier   \
                      --rehearsal   