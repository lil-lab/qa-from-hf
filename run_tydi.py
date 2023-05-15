# Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved
"""Run BERT on Tydi.


GPU-low
for training
python run_tydi.py \
  --do_train \
  --do_eval \
  --model microsoft/deberta-base \
  --train_file data/tydiqa-v1.0-train.jsonl.gz \
  --dev_file data/tydiqa-v1.0-dev.jsonl.gz \
  --train_batch_size 40 \
  --gradient_accumulation_steps 8 \
  --eval_batch_size 5  \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --eval_per_epoch 800 \
  --max_seq_length 512 \
  --doc_stride 512 \
  --eval_metric f1 \
  --na_prob_thresh 0 \
  --output_dir tydi_output \
  --version_2_with_negative


for testing 
python run_tydi.py \
  --do_eval \
  --eval_test \
  --model microsoft/deberta-base \
  --test_file data/tydiqa-v1.0-train.jsonl.gz \
  --eval_batch_size 5  \
  --max_seq_length 512 \
  --doc_stride 512 \
  --eval_metric f1 \
  --na_prob_thresh 0 \
  --initialize_model_from_checkpoint tydi_output/20220607135748982502 \
  --output_dir tydi_output \
  --version_2_with_negative
"""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import math
import os
import random
import time
import re
import string
from io import open
import gzip
import datetime

from IPython import embed

import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from transformers import BertTokenizer, DebertaTokenizer, DebertaTokenizerFast, DebertaV2TokenizerFast
from transformers import AdamW
from model import BertForQuestionAnswering
from transformers import get_scheduler
from datasets import load_dataset

from pytorch_pretrained_bert.tokenization import BasicTokenizer  # used in evaluation

from pytorch_pretrained_bert.tokenization import whitespace_tokenize
from src.data import get_nq_data, get_tydi_data, read_squad_examples_and_features, read_tydi_examples_and_features, get_mrqa_data, read_mrqa_examples_and_features
from src.eval import evaluate

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

PRED_FILE = "predictions.json"
EVAL_FILE = "eval_results.txt"
TEST_FILE = "test_results.txt"



# https://github.com/google-research-datasets/tydiqa/blob/43cde6d598c1cf88c1a8b9ed32e89263ffb5e03b/tydi_eval.py#L239
# return a string
def byte_slice(text, start, end):
    byte_str = bytes(text, 'utf-8')
    # return str(byte_str[start:end])
    return byte_str[start:end].decode('utf-8')


def normalize_answer(s):
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def load_initialization(model, args):
    ckpt = torch.load(args.initialize_model_from_checkpoint + '/saved_checkpoint')
    assert args.model == ckpt['args']['model'], args.model + ' vs ' + ckpt['args']['model']
    model.load_state_dict(ckpt['model_state_dict'])
    logger.info("***** Model Initialization *****")
    logger.info("Loaded the model state from a saved checkpoint {}".format(
        args.initialize_model_from_checkpoint))


def main(args):
    print('rand seed', int(args.random_indices.split('.')[0].split('_')[-1]))
    # create timestamp: folder name, wandb logging
    args.timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # NOTE not testable yet for n-gpu training
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}, 16-bits training: {}".format(device, n_gpu, args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    args.train_batch_size = \
        args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.do_train:
        assert (args.train_file is not None) and (args.dev_file is not None)

    if args.eval_test:
        assert args.test_file is not None
    else:
        assert args.dev_file is not None

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # set up the logging for this experiment: create a folder named by the timestamp
    # args.output_dir += '/' + args.timestamp
    if args.do_train:
        # set up the logging for this experiment: create a folder named by hyperparameters 
        # also, during eval, the folder need not be created again
        args.output_dir += '/%s_initial_data_%d_wclassifier_%d/'%(args.data_type, args.num_initial_data, int(args.random_indices.split('.')[0].split('_')[-1])) 
        os.makedirs(args.output_dir)
        logger.info('output_dir: %s'%(args.output_dir))
        # args.output_dir += '/test'

    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(args)

    if 'deberta' in args.model:
        tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model, return_offsets_mapping=True)
    elif 'bert-' in args.model:
        tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
    else:
        raise ValueError('Model type!')


    if args.do_train:
        if args.data_type == 'tydi':
            # for training 
            train_dataset = get_tydi_data(args.train_file)
            eval_dataset = get_tydi_data(args.dev_file)

            unfiltered_train_examples, unfiltered_train_features = read_tydi_examples_and_features(
                input_data=train_dataset,
                is_training=True,
                version_2_with_negative=args.version_2_with_negative,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                prepend_title=args.prepend_title
                )

            eval_examples, eval_features = read_tydi_examples_and_features(
                input_data=eval_dataset,
                is_training=False,
                version_2_with_negative=args.version_2_with_negative,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                prepend_title=args.prepend_title
                )
        elif args.data_type == 'squad2':
            logger.info('loading from squad_v2...')
            input_data = load_dataset("squad_v2")
            
            unfiltered_train_examples, unfiltered_train_dataset, unfiltered_train_features = read_squad_examples_and_features(
                is_training=True,
                version_2_with_negative=True,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                prepend_title=args.prepend_title,
                input_data=input_data['train']
                )

            eval_examples, eval_dataset, eval_features = read_squad_examples_and_features(
                is_training=False,
                version_2_with_negative=True,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                prepend_title=args.prepend_title,
                input_data=input_data['validation']
                )
        elif args.data_type == 'squad':
            logger.info('loading from squad...')
            input_data = load_dataset("squad")
            
            unfiltered_train_examples, _, unfiltered_train_features = read_squad_examples_and_features(
                is_training=True,
                version_2_with_negative=False,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                prepend_title=args.prepend_title,
                input_data=input_data['train']
                )

            eval_examples, eval_dataset, eval_features = read_squad_examples_and_features(
                is_training=False,
                version_2_with_negative=False,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                prepend_title=args.prepend_title,
                input_data=input_data['validation']
                )
        elif args.data_type == 'newsqa' or args.data_type == 'searchqa' or args.data_type == 'triviaqa':
            logger.info('loading from newsqa...')
            train_dataset = get_mrqa_data(args.train_file)
            eval_dataset = get_mrqa_data(args.dev_file)

            unfiltered_train_examples, _,  unfiltered_train_features = read_mrqa_examples_and_features(
                input_data=train_dataset,
                is_training=True,
                version_2_with_negative=False,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                prepend_title=args.prepend_title
                )

            eval_examples, _, eval_features = read_mrqa_examples_and_features(
                input_data=eval_dataset,
                is_training=False,
                version_2_with_negative=False,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                prepend_title=args.prepend_title
                )


        if args.upsample:
            # if do upsample, then we want (answerable: unanswerable = 3:1)
            # so we make the number of answerable examples 7x, and keep the unanswerable ones
            train_examples = []
            train_features = []
            num_unanswerable = 0
            for i, l in enumerate(unfiltered_train_examples):
                if l.start_sample == -1:
                    train_examples.append(l)
                    train_features.append(unfiltered_train_features[i])
                    num_unanswerable += 1
                else :
                    for _ in range(7):
                        train_examples.append(l)
                        train_features.append(unfiltered_train_features[i])
            print('num_unanswerable', num_unanswerable)
            print('train_examples', len(train_examples))
        else:
            train_examples = unfiltered_train_examples
            train_features = unfiltered_train_features
            train_dataset = unfiltered_train_dataset

        if args.num_initial_data:  # vary the number of initial data used
            random_indices = [int(l) for l in open(args.random_indices)]
            assert len(random_indices) == len(train_features)
            train_examples = [train_examples[i] for i in random_indices]
            train_features = [train_features[i] for i in random_indices]
            train_dataset = [train_dataset[i] for i in random_indices]
            train_examples = train_examples[:args.num_initial_data]
            train_features = train_features[:args.num_initial_data]
            train_dataset = train_dataset[:args.num_initial_data]
            
            unans = 0
            for f in train_features:
                if f.start_sample == -1 and f.end_sample == -1:
                    unans += 1
            logger.info('Num UnAns in Random Subset: %d / %d'%(unans, len(train_features)))

            eval_examples = train_examples
            eval_features = train_features
            eval_dataset = train_dataset

        if args.add_squad_examples:
            # add squad examples to the training set 
            squad_examples, _, squad_features = read_squad_examples_and_features(
                is_training=True,
                version_2_with_negative=False,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                prepend_title=args.prepend_title
                )
            
            random_indices = list(range(len(squad_examples)))
            random.shuffle(random_indices)
            squad_examples = [squad_examples[i] for i in random_indices]
            squad_features = [squad_features[i] for i in random_indices]
            
            # add different number of SQuAD examples
            squad_examples = squad_examples[:len(train_examples)]
            squad_features = squad_features[:len(train_examples)]

            train_examples += squad_examples
            train_features += squad_features



        logger.info("***** Dev *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)
        for i, f in enumerate(eval_features):
            f.example_index = i
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)



        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_sample for f in train_features],
                                           dtype=torch.long)
        all_end_positions = torch.tensor([f.end_sample for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = \
            len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        logger.info("***** Train *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        if args.num_initial_data:
            logger.info("  Num Initial Data = %d", args.num_initial_data)

        eval_step = max(1, len(train_batches) // args.eval_per_epoch)
        best_result = None
        # NOTE only tested for one learning rate
        lrs = [args.learning_rate] if args.learning_rate else \
            [1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
        for lr in lrs:
            # NOTE old: model = BertForQuestionAnswering.from_pretrained(args.model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
            model = BertForQuestionAnswering(model_type=args.model)
            if args.fp16:
                model.half()

            if args.add_classifier:
                model.classification = nn.Linear(model.bert.config.hidden_size, 2)
            model.to(device)
            


            if n_gpu > 1:
                model = torch.nn.DataParallel(model)
            param_optimizer = list(model.named_parameters())
            param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [{
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay':
                0.01
            }, {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay':
                0.0
            }]

            optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
            lr_scheduler = get_scheduler(args.scheduler,
                                         optimizer=optimizer,
                                         num_warmup_steps=int(num_train_optimization_steps *
                                                              args.warmup_proportion),
                                         num_training_steps=num_train_optimization_steps)

            if args.wandb:
                wandb.init(
                    project='',
                    entity='',
                    name=f'{args.model}_{args.scheduler}={lr}_b{args.train_batch_size}_ep{args.num_train_epochs}/{args.timestamp}',
                    notes=args.notes,
                    config=vars(args),
                    tags=['title_seg=2', 'initial-v3-base', args.data_type, 'squad2_initial_data_256_wclassifier'])
                wandb.watch(model)
# 
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0
            global_step = 0
            max_valid_em = 0
            max_valid_f1 = 0
            start_time = time.time()
            for epoch in range(int(args.num_train_epochs)):
                model.train()
                logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
                if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                    random.shuffle(train_batches)
                for step, batch in enumerate(train_batches):
                    if n_gpu == 1:
                        batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                    loss, classifier_prob = model(batch, classifier=args.add_classifier)
                    if n_gpu > 1:
                        loss = loss.mean()

                    

                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                        if args.wandb:
                            wandb.log(
                                {
                                    'Train Loss': loss * args.gradient_accumulation_steps,
                                }, step=global_step)
                            if args.add_classifier:
                                classifier_pred = classifier_prob.argmax(dim=-1)
                                answerable_mask = (start_positions != 0) | (end_positions != 0)
                                acc = 100 * (classifier_pred == answerable_mask).long().sum()/float(classifier_pred.size(0))

                                wandb.log(
                                    {
                                        'Pert. Unans': 100 * (1-classifier_pred).sum() / float(classifier_pred.size(0)),
                                        'classification_acc': acc,
                                    }, step=global_step)

                    if (step + 1) % eval_step == 0:
                        logger.info(
                            'Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                                epoch, step + 1, len(train_batches),
                                time.time() - start_time, tr_loss / nb_tr_steps))

                        save_model = False
                        if args.do_eval:
                            result, _, _, preds = \
                                evaluate(args, model, device, eval_dataset,
                                         eval_dataloader, eval_examples, eval_features, 
                                         args.na_prob_thresh, tokenizer, args.data_type,
                                         calculate_score=args.calculate_score, classifier=args.add_classifier)
                            model.train()

                            if args.calculate_score:
                                result['global_step'] = global_step
                                result['epoch'] = epoch
                                result['learning_rate'] = lr
                                result['batch_size'] = args.train_batch_size
                                
                                if (best_result is None) or (result[args.eval_metric] >
                                                             best_result[args.eval_metric]):
                                    best_result = result
                                    save_model = True
                                    logger.info(
                                        "!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                        (args.eval_metric, str(lr), epoch, result[args.eval_metric]))

                                max_valid_em = max(max_valid_em, result['exact'])
                                max_valid_f1 = max(max_valid_f1, result['f1'])
                                if args.wandb:
                                    wandb.log(
                                                {
                                                    '(Valid) F1': result['f1'],
                                                    '(Valid) Exact': result['exact'],
                                                    '(Valid) Max F1': max_valid_f1,
                                                    '(Valid) Max Exact': max_valid_em,
                                                }, step=global_step
                                            )
                            else:
                                save_model = True
                        else:
                            save_model = True
                        if save_model:
                            if n_gpu > 1:
                                # save the config
                                model.module.bert.config.to_json_file(
                                    os.path.join(args.output_dir, 'config.json'))
                                # save the model
                                torch.save(
                                    {
                                        'global_step': global_step,
                                        'args': vars(args),
                                        'model_state_dict': model.module.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                    }, os.path.join(args.output_dir, 'saved_checkpoint'))
                            else:

                                # save the config
                                model.bert.config.to_json_file(
                                    os.path.join(args.output_dir, 'config.json'))
                                # save the model
                                torch.save(
                                    {
                                        'global_step': global_step,
                                        'args': vars(args),
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                    }, os.path.join(args.output_dir, 'saved_checkpoint'))
                            if best_result:
                                # i.e. best_result is not None
                                filename = EVAL_FILE
                                if len(lrs) != 1:
                                    filename = str(lr) + '_' + EVAL_FILE
                                with open(os.path.join(args.output_dir, filename), "w") as writer:
                                    for key in sorted(best_result.keys()):
                                        writer.write("%s = %s\n" % (key, str(best_result[key])))
                                    if epoch == 0:
                                        one_epoch_f1 = best_result['f1']
                                    writer.write("%s = %s\n" % ('one_epoch_f1', one_epoch_f1))

    if args.do_eval:
        if args.eval_test:
            if not args.test_on_squad:
                # test on TyDiQA data
                eval_dataset = get_tydi_data(args.test_file)
                eval_examples, eval_features = read_tydi_examples_and_features(
                    input_data=eval_dataset,
                    is_training=False,
                    version_2_with_negative=args.version_2_with_negative,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    prepend_title=args.prepend_title
                    )
            else:
                # test on SQuAD data
                print('test on squad....')
                eval_examples, eval_dataset, eval_features = read_squad_examples_and_features(
                # eval_examples, eval_dataset,  eval_features = read_squad_examples_and_features(
                    is_training=False,
                    version_2_with_negative=False,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    prepend_title=args.prepend_title
                    )

            logger.info("***** Test *****")
            logger.info("  Num orig examples = %d", len(eval_examples))
            logger.info("  Num split examples = %d", len(eval_features))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                      all_example_index)
            eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
        # NOTE old: model = BertForQuestionAnswering.from_pretrained(args.output_dir)
        # model = BertForQuestionAnswering(model_type=args.model)
        # NOTE change: only evaluate on the test set
        if not args.do_train:
            model = BertForQuestionAnswering(model_type=args.model)
            load_initialization(model=model, args=args)
        if args.fp16:
            model.half()
        model.to(device)

        na_prob_thresh = args.na_prob_thresh
        if args.version_2_with_negative:
            eval_result_file = os.path.join(args.output_dir, "eval_results.txt")
            if os.path.isfile(eval_result_file):
                with open(eval_result_file) as f:
                    for line in f.readlines():
                        if line.startswith('best_f1_thresh'):
                            na_prob_thresh = float(line.strip().split()[-1])
                            logger.info("na_prob_thresh = %.6f" % na_prob_thresh)

        result, _, _, preds = \
            evaluate(args, model, device, eval_dataset,
                     eval_dataloader, eval_examples, eval_features,
                     na_prob_thresh=na_prob_thresh, tokenizer=tokenizer, dataset_name=args.data_type,
                     calculate_score=True, classifier=args.add_classifier)
        with open(os.path.join(args.output_dir, PRED_FILE), "w") as writer:
            writer.write(json.dumps(preds, indent=4) + "\n")
        with open(os.path.join(args.output_dir, TEST_FILE), "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--train_file",
                        default=None,
                        type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--dev_file",
                        default=None,
                        type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--eval_per_epoch",
                        default=10,
                        type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride",
                        default=128,
                        type=int,
                        help="When splitting up a long document into chunks, "
                        "how much stride to take between chunks.")
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_test",
                        action='store_true',
                        help='Wehther to run eval on the test set.')
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for predictions.")
    parser.add_argument("--learning_rate",
                        default=None,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--eval_metric", default='f1', type=str)
    parser.add_argument("--train_mode",
                        type=str,
                        default='random_sorted',
                        choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
        "of training.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json "
        "output file.")
    parser.add_argument("--max_answer_length",
                        default=512,
                        type=int,
                        help="The maximum length of an answer that can be generated. "
                        "This is needed because the start "
                        "and end predictions are not conditioned on one another.")
    parser.add_argument(
        "--verbose_logging",
        action='store_true',
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument(
        '--loss_scale',
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')

    # below are added
    parser.add_argument('--scheduler', default='linear', type=str, help='Learning rate scheduler.')
    parser.add_argument('--na_prob_thresh',
                        type=float,
                        default=0.0,
                        help='0.0 means no threshholding for na probs')
    parser.add_argument('--initialize_model_from_checkpoint',
                        default=None,
                        help='Relative filepath to a saved checkpoint as model initialization.')
    parser.add_argument('--add_squad_examples',
                        action='store_true',
                        help='If true, add squad examples to balance unanswerable (do data augmentation)')
    parser.add_argument('--upsample',
                        action='store_true',
                        help='whether to upsample answerable data')
    parser.add_argument('--dev_fewer_unans',
                        action='store_true',
                        help='whether to include fewer unanswerable examples for dev set')
    parser.add_argument('--test_on_squad',
                        action='store_true',
                        help='whether test on SQuAD Dev data')
    parser.add_argument('--prepend_title',
                        action='store_true',
                        help='whether to prepend the document title to the document text before tokenization')
    parser.add_argument('--squad_file', default='data/squad_train.json', type=str, help='squad train file path (for add_squad_exapmles) for data augmentation, not for evaluation')
    parser.add_argument('--wandb',
                        action='store_true',
                        help='If true, log with wandb.')
    parser.add_argument('--notes', default='', type=str, help='notes added for wandb')
    parser.add_argument('--num_initial_data', default=None, type=int, help='number of init data')
    parser.add_argument('--data_type', default='tydi', type=str, help='type of input data')
    parser.add_argument('--random_indices', default='', type=str, help='random indices for dataset')
    parser.add_argument('--add_classifier', action='store_true', help='whether to add a separate classifier for unans/ans')
    parser.add_argument('--calculate_score', action='store_true', help='whether to add a separate classifier for unans/ans')

    args = parser.parse_args()

    main(args)

