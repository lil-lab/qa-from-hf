from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import os
import random
import time
from io import open
import datetime
from tqdm import trange

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from transformers import BertTokenizer, DebertaTokenizerFast, DebertaV2TokenizerFast, AutoTokenizer
from transformers import AdamW
from model import BertForQuestionAnsweringSequence, BertForQuestionAnswering, DebertaSQuAD2
from transformers import get_scheduler, get_cosine_with_hard_restarts_schedule_with_warmup
from datasets import load_dataset

import wandb
from prettytable import PrettyTable

from src.eval import evaluate
from src.data import get_feedback_data, get_nq_data, get_tydi_data, read_feedback_examples_and_features, read_squad_examples_and_features, read_tydi_examples_and_features, get_mrqa_data, read_mrqa_examples_and_features

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

PRED_FILE = "predictions-train.json"
EVAL_FILE = "eval_results.txt"
TEST_FILE = "test_results.txt"
CSV_FILE = "results_sheet_test_all.tsv"
PLOT_CSV_FILE = "plot_per_round.tsv"

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def turn_off_dropout(m):
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0


def load_initialization(model, args):
    ckpt = torch.load(args.initialize_model_from_checkpoint + '/' + args.checkpoint_name)
    assert args.model == ckpt['args']['model'], args.model + ' vs ' + ckpt['args']['model']
    model.load_state_dict(ckpt['model_state_dict'])
    logger.info("***** Model Initialization *****")
    logger.info("Loaded the model state from a saved checkpoint {}".format(
        args.initialize_model_from_checkpoint))


def get_log_probs(start_probs, end_probs, start_positions, end_positions, args,
                      device):

    start_samples, end_samples = start_positions, end_positions
    ignored_index = start_probs.size(1)
    start_samples.clamp_(0, ignored_index)
    end_samples.clamp_(0, ignored_index)

    bs = start_samples.shape[0]
    log_probs = start_probs[torch.arange(bs), start_samples].log() + end_probs[torch.arange(bs),
                                                                              end_samples].log()
    return log_probs


def collect_rewards_offline(model, train_batches, args, device, tokenizer, n_gpu, is_initial=False):
    total_pos = 0
    total_neg = 0

    for i in trange(len(train_batches)):
        batch = train_batches[i]
        batch = tuple(t.to(device) for t in batch)

        if args.load_log_prob:
            input_ids, input_mask, segment_ids, start_samples, end_samples, class_samples, log_probs, class_log_probs, rewards, class_rewards = batch
        else:
            input_ids, input_mask, segment_ids, start_samples, end_samples, class_samples, rewards, class_rewards = batch
            if is_initial:
                log_probs = torch.zeros(rewards.size()).to(device)
                class_log_probs = torch.zeros(rewards.size()).to(device)
            else:
                with torch.no_grad():
                    start_probs, end_probs, class_probs = model(batch=batch[:3], return_prob=True)
                    log_probs = get_log_probs(start_probs, end_probs, start_samples,
                                                      end_samples, args, device)
                    if args.add_classifier:
                        class_log_probs = class_probs[torch.arange(bs), class_samples].log()
                    else:
                        class_log_probs = None
            train_batches[i] = [
                input_ids, input_mask, segment_ids, start_samples, end_samples, class_samples, log_probs, class_log_probs, rewards, class_rewards
            ]

        count_pos = torch.sum(rewards > 0).item()
        total_pos += count_pos
        total_neg += input_ids.shape[0] - count_pos
    return train_batches, total_pos, total_neg


def prepare_data(args, filename, tokenizer, data_type, batch_size, data_split='train'):
    if data_type == 'feedback':
        logger.info('loading feedback data (%s)...' % (data_split))
        dataset = get_feedback_data(filename)
        examples, features = read_feedback_examples_and_features(dataset,
                                                                 args.negative_reward,
                                                                 args.partial_reward,
                                                                 args.reward_wrong_unans,
                                                                 args.reward_correct_span,
                                                                 args.reward_correct_unans,
                                                                 args.reward_class_wrong,
                                                                 args.reward_class_correct_ans,
                                                                 tokenizer,
                                                                 args.max_seq_length,
                                                                 args.prepend_title,
                                                                 load_log_prob=args.load_log_prob
                                                                 and data_split == 'train')
    elif data_type == 'tydi':
        logger.info('loading from tydi..., is_training = %s' % (str((data_split == 'train'))))
        dataset = get_tydi_data(filename)
        examples, features = read_tydi_examples_and_features(
            input_data=dataset,
            is_training=(data_split == 'train'),
            version_2_with_negative=args.version_2_with_negative,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            prepend_title=args.prepend_title)
    elif data_type == 'squad':
        logger.info('loading from squad..., is_training = %s' % (str((data_split == 'train'))))
        input_data = load_dataset("squad")
        is_training = (data_split == 'train')
        if is_training:
            input_data = input_data['train']
        else:
            input_data = input_data['validation']
        examples, dataset, features = read_squad_examples_and_features(
            is_training=(data_split == 'train'),
            version_2_with_negative=False,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            prepend_title=args.prepend_title,
            input_data=input_data,
        )
    elif data_type == 'squad2':
        logger.info('loading from squad_v2..., is_training = %s' % (str((data_split == 'train'))))
        input_data = load_dataset("squad_v2")
        is_training = (data_split == 'train')
        if is_training:
            input_data = input_data['train']
        else:
            input_data = input_data['validation']
        examples, dataset, features = read_squad_examples_and_features(
            is_training=(data_split == 'train'),
            version_2_with_negative=True,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            prepend_title=args.prepend_title,
            input_data=input_data)
    elif data_type == 'nq':
        logger.info('loading from NQ..., is_training = %s' % (str((data_split == 'train'))))
        dataset = get_nq_data(filename)
        assert data_split != 'train'
        examples, _, features = read_squad_examples_and_features(
            is_training=(data_split == 'train'),
            version_2_with_negative=False,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            prepend_title=args.prepend_title,
            get_dataset=False,
            input_data=dataset)
    elif data_type == 'tydi+squad':
        assert (data_split == 'train')
        dataset = get_tydi_data(filename)
        examples, features = read_tydi_examples_and_features(
            input_data=dataset,
            is_training=True,
            version_2_with_negative=args.version_2_with_negative,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            prepend_title=args.prepend_title)
        squad_examples, _, squad_features = read_squad_examples_and_features(
            is_training=True,
            version_2_with_negative=False,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            prepend_title=args.prepend_title)

        random_indices = [int(l.strip('\n')) for l in open('random_indices_squad.txt')]
        squad_examples = [squad_examples[i] for i in random_indices]
        squad_features = [squad_features[i] for i in random_indices]

        # add SQuAD examples
        examples += squad_examples[:len(examples)]
        features += squad_features[:len(features)]

    elif data_type == 'newsqa' or data_type == 'searchqa' or data_type == 'triviaqa':
        logger.info('loading from mrqa...')
        dataset = get_mrqa_data(filename)

        examples, dataset, features = read_mrqa_examples_and_features(
            input_data=dataset,
            is_training=(data_split == 'train'),
            version_2_with_negative=False,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            prepend_title=args.prepend_title,
            get_dataset=True
            )

    # shuffle the data
    if data_split == 'train':
        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            features = sorted(features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(features)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    if data_split == 'train':
        all_start_samples = torch.tensor([f.start_sample for f in features], dtype=torch.long)
        all_end_samples = torch.tensor([f.end_sample for f in features], dtype=torch.long)
        all_class_samples = torch.tensor([f.class_sample for f in features], dtype=torch.long)
        all_rewards = torch.tensor([f.reward for f in features], dtype=torch.float)
        all_class_rewards = torch.tensor([f.class_reward for f in features], dtype=torch.float)

        if args.load_log_prob:
            all_log_probs = torch.tensor([f.log_prob for f in features], dtype=torch.float)
            all_class_log_probs = torch.tensor([f.class_log_prob for f in features], dtype=torch.float)
            data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_samples,
                                 all_end_samples, all_class_samples, all_log_probs, all_class_log_probs, all_rewards, all_class_rewards)
        else:
            data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_samples,
                                 all_end_samples, all_class_samples, all_rewards, all_class_rewards)
        logger.info("***** Train *****")
        logger.info("  Num examples = %d", len(features))
        logger.info("  Batch size = %d", batch_size)
    else:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
        logger.info("***** %s *****" % (data_split))
        logger.info("  Num orig examples = %d", len(examples))
        logger.info("  Num split examples = %d", len(features))
        logger.info("  Batch size = %d", batch_size)

    dataloader = DataLoader(data, batch_size=batch_size)
    batches = [batch for batch in dataloader]
    return dataset, examples, features, dataloader, batches


def fetch_batch_data(step, train_batches):
    if step % len(train_batches) == 0:
        logger.info("shuffling previous data...")
        random.shuffle(train_batches)
    return train_batches[step % len(train_batches)]


def main(args):
    # create timestamp: folder name, wandb logging
    args.timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu
    # random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # argparse checkers
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    if args.do_train:
        assert (args.train_file is not None) and (args.dev_file is not None)
    if args.eval_test:
        assert args.test_file is not None
    else:
        assert args.dev_file is not None
    # only evaluate on the test set: need an initialization
    # if args.eval_test and not args.do_train:
    #     assert args.initialize_model_from_checkpoint is not None

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_train:
        # set up the logging for this experiment: create a folder named by hyperparameters
        # also, during eval, the folder need not be created again
        # we use the same folder for storing eval results
        # set up the logging for this experiment: create a folder named by the timestamp
        model_name = args.model.split('/')[-1]
        args.output_dir += '/' + f'round{args.round_index}/rehearsal_round{args.round_index}_{model_name}_{args.scheduler}_{args.learning_rate}_b{args.train_batch_size}_acc{args.gradient_accumulation_steps}_ep{args.num_train_epochs}_nr{args.negative_reward}_pr{args.partial_reward}_rcu{args.reward_correct_unans}/{args.timestamp}'
        os.makedirs(args.output_dir)
        # args.output_dir += '/test'

    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(args)

    if args.model == "deepset/deberta-v3-base-squad2":
        tokenizer = AutoTokenizer.from_pretrained(args.model, return_offsets_mapping=True)
    elif args.model == 'microsoft/deberta-v3-base':
        tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model, return_offsets_mapping=True)
    else:
        raise ValueError('Model type!')

    
    if args.do_train:
        # multiple dataloaders (current round, all previous rounds, initial data)
        all_train_batches = []
        # for training

        ########## Data Preparation Begins ########
        file_list = [l.strip('\n') for l in open(args.train_file)]
        print('file len = %d' % len(file_list))
        # handle batch size in different batch
        if args.rehearsal:
            count = len(file_list)
            train_batch_sizes = [int(args.train_batch_size / count)] * count
        else:
            train_batch_sizes = [args.train_batch_size]

        print(args.train_batch_size, train_batch_sizes)
        assert sum(train_batch_sizes) == args.train_batch_size

        initial_train_dataloader = None
        for i, file_ in enumerate(file_list):
            _, _, train_features, train_dataloader, train_batches = prepare_data(args=args,
                                                                    filename=file_,
                                                                    tokenizer=tokenizer,
                                                                    data_type='feedback',
                                                                    batch_size=train_batch_sizes[i],
                                                                    data_split='train')
            if i == 0:
                logger.info('Reading Current Data From %s' % file_)
                initial_train_dataloader = train_dataloader
            else:
                logger.info('Reading Previous Data From %s' % file_)
            all_train_batches.append(train_batches)

        num_train_optimization_steps = (len(initial_train_dataloader) //
                                        args.gradient_accumulation_steps) * args.num_train_epochs
        logger.info("  Num steps = %d | len current round: %d" %
                    (num_train_optimization_steps, len(all_train_batches[0])))

        # validation dataset
        eval_dataset, eval_examples, eval_features, eval_dataloader, _ = prepare_data(
            args=args,
            filename=args.dev_file,
            tokenizer=tokenizer,
            data_type=args.valid_data_type,
            batch_size=args.eval_batch_size,
            data_split='valid')

        ########## Data Preparation Ends ########
        assert len(all_train_batches) == len(train_batch_sizes)
        eval_step = max(1, len(all_train_batches[0]) // args.eval_per_epoch)
        logger.info('Time_Stamp %s ' % args.timestamp + 'eval step: %d' % eval_step)

        # NOTE only tested for one learning rate
        assert args.learning_rate
        lr = args.learning_rate

        if args.model == "deepset/deberta-v3-base-squad2":
            model = DebertaSQuAD2(model_type=args.model)
            print('loading deepset model')
        else:
            if args.add_classifier:
                model = BertForQuestionAnsweringSequence(model_type=args.model)
            else:
                model = BertForQuestionAnswering(model_type=args.model)

        # initial from pretraining  # initialize model; no matter training or test
        if args.initialize_model_from_checkpoint:
            load_initialization(model, args)

        if args.turn_off_dropout:
            turn_off_dropout(model)

        model.to(device)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # for setting up loss visualization
        if args.wandb:
            wandb.init(
                project="",
                entity='',
                name=
                f'round{args.round_index}_{args.model}_{args.scheduler}={lr}_b{args.train_batch_size}_ep{args.num_train_epochs}_nr{args.negative_reward}_pr{args.partial_reward}/{args.timestamp}',
                notes=args.notes,
                config=vars(args),
                tags=[
                    args.tag, 'main experiment - classifier', '200-ex',
                    'round %d' % args.round_index,
                    'w/ rehearsal' if args.rehearsal else 'w/o rehearsal',
                    ':'.join([str(l) for l in train_batch_sizes]), 'max_answer_length=30',
                    'squad2.0 - 512 initial',
                    'correct_unans=%f' % args.reward_correct_unans, 'hyperparameter', args.timestamp, '30% unans', 'class_coeff=%2.2f'%(args.class_coeff),
                    'entropy_coeff=%2.2f'%(args.entropy_coeff), 'main_task', 'reward_class_wrong=%2.2f'%(args.reward_class_wrong)
                ])
            wandb.watch(model)

        file_list = [l.strip('\n') for l in open(args.train_file)]
        for f_ in file_list:
            logger.info("reading from file: %s" % f_)

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

        # for offline training
        if args.setup == 'offline':
            for i in range(len(all_train_batches)):  # collect reward for each dataloader
                all_train_batches[i], total_pos, total_neg = collect_rewards_offline(
                    model, all_train_batches[i], args, device, tokenizer, n_gpu)
                logger.info("Offline regret computation: {} positives {} negatives".format(
                    total_pos, total_neg))

        # start training
        best_result = None
        tr_loss = 0
        nb_tr_steps = 0
        num_train_batches = 0
        global_step = 0
        max_valid_reward = 0
        max_valid_f1 = 0
        max_valid_em = 0
        start_time = time.time()
        simulation_log = None
        one_epoch_f1 = None

        for epoch in range(int(args.num_train_epochs)):
            rewards_per_epoch = []
            class_rewards_per_epoch = []
            acc_per_epoch = 0
            model.train()
            logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
            if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                logger.info("shuffling the main training batches")
                random.shuffle(all_train_batches[0])

            for step, initial_batch in enumerate(all_train_batches[0]):
                # prepare batch for rehearsal
                # use the main (current round) training set to keep track of num_training_epochs
                sizes = [initial_batch[0].size(0)]
                all_batches = [[t] for t in initial_batch]
                for loader_index in range(1, len(all_train_batches)):
                    # fetch data from dataloader
                    partial_batch = fetch_batch_data(num_train_batches,
                                                     all_train_batches[loader_index])
                    assert (len(initial_batch) == len(partial_batch)) and (len(all_batches)
                                                                           == len(partial_batch))
                    sizes.append(partial_batch[0].size(0))
                    # add data from loader to the current batch
                    for j in range(len(partial_batch)):
                        all_batches[j].append(partial_batch[j])


                batch = [torch.cat(all_batches[j], dim=0) for j in range(len(all_batches))]
                batch = tuple(t.to(device) for t in batch)
                num_train_batches += 1

                
                ########## do Bandit Learning #########
                start_probs, end_probs, class_probs = model(batch=batch[:3], return_prob=True, classifier=args.add_classifier)

                bs = start_probs.shape[0]
                if args.setup == 'online':  # should not use online
                    input_ids, _, _, start_samples, end_samples, rewards = batch
                    start_samples, end_samples, log_prob, rewards = get_batch_rewards(
                        start_probs, end_probs, start_samples, end_samples, rewards, args, device)
                    count_pos = torch.sum(rewards > 0).item()
                    total_pos += count_pos
                    total_neg += bs - count_pos
                else:
                    input_ids, _, _, start_samples, end_samples, class_samples, old_log_probs, old_class_log_probs, old_rewards, old_class_rewards = batch

                    ignored_index = start_probs.size(1)
                    start_samples.clamp_(0, ignored_index)
                    end_samples.clamp_(0, ignored_index)

                    log_probs = start_probs[torch.arange(bs),
                                           start_samples].log() + end_probs[torch.arange(bs),
                                                                            end_samples].log()
                    ratios = torch.exp(log_probs - old_log_probs)
                    rewards = torch.clamp(ratios, 0, 1) * old_rewards
                    rewards = rewards.detach()

                    if args.add_classifier:
                        class_log_probs = class_probs[torch.arange(bs), class_samples].log()
                        class_ratios = torch.exp(class_log_probs - old_class_log_probs)
                        class_rewards = torch.clamp(class_ratios, 0, 1) * old_class_rewards
                        class_rewards = class_rewards.detach()

                        class_pred = class_probs.argmax(dim=-1)
                        acc = ((class_samples == class_pred) == (old_class_rewards > 0)).long().sum()
                        # print('acc', acc)
                        acc_per_epoch += acc
                    
            
                rewards_per_epoch.append(rewards.mean().item())
                if args.add_classifier:
                    class_rewards_per_epoch.append(class_rewards.mean().item())

                ########## Update Model ###########
                detached_advantages = rewards
                loss = (-log_probs * detached_advantages) / 2

                if args.add_classifier:
                    class_detached_advantages = class_rewards
                    class_loss = (-class_log_probs * class_detached_advantages)
                    classifier_entropy = torch.mean(torch.sum(-class_probs * class_probs.log(), dim=-1))
                    loss = loss + args.class_coeff * class_loss - args.entropy_coeff * classifier_entropy
                else:
                    class_loss = torch.zeros((1,))
                    classifier_entropy = torch.zeros((1,))

                loss = loss.mean()



                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                tr_loss += loss.item()
                nb_tr_steps += 1

                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                if args.wandb and (global_step + 1) % 5 == 0:
                    wandb.log(
                        {
                            '(Train) batch policy loss': loss.item(),
                            '(Train) Span loss': (-log_probs * detached_advantages / 2).mean().item(),
                            '(Train) classification loss': class_loss.mean().item(),
                            '(Train) batch advantage': detached_advantages.mean().item(),
                            'IPS ratios':
                            torch.clamp(ratios, 0, 1).mean().item(),
                            '(Train) cls entropy': classifier_entropy.item(),

                        },
                        step=global_step)
                    if args.add_classifier:
                        wandb.log(
                            {
                            '(Train) batch classification advantage': class_detached_advantages.mean().item(),
                            '(Train) batch class advantage':
                            ((class_detached_advantages).sum() / class_detached_advantages.size(0)).item(),
                            '(Train) class_log_probs':
                            class_log_probs.mean().item(),
                            'IPS ratios of classification':
                            torch.clamp(class_ratios, 0, 1).mean().item(),
                            }, step=global_step)
                    if simulation_log is not None:
                        wandb.log(simulation_log, step=global_step)


                if step != 0 and (step) % eval_step == 0:
                    ## record training related info
                    logger.info(
                        'Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                            epoch, step + 1, len(all_train_batches[0]),
                            time.time() - start_time, tr_loss / (nb_tr_steps+1)))

                    tr_loss = 0
                    nb_tr_steps = 0

                    save_model = False

                    ######## validation ########
                    if args.do_eval:
                        result, has_ans_eval, no_ans_eval, _ = \
                            evaluate(args, model, device, eval_dataset,
                                     eval_dataloader, eval_examples, eval_features, 
                                     args.na_prob_thresh, tokenizer, args.valid_data_type, 
                                     calculate_score=True, classifier=args.add_classifier)
                        model.train()
                        result['global_step'] = global_step
                        result['epoch'] = epoch
                        result['learning_rate'] = lr
                        result['batch_size'] = args.train_batch_size

                        if global_step > 1 and ((best_result is None) or (result[args.eval_metric] >
                                                     best_result[args.eval_metric])):
                            best_result = result
                            save_model = True
                            logger.info(
                                "!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                (args.eval_metric, str(lr), epoch, result[args.eval_metric]))

                        # record max f1, em, reward
                        max_valid_em = max(max_valid_em, result['exact'])
                        max_valid_f1 = max(max_valid_f1, result['f1'])
                        if args.valid_data_type == 'feedback':
                            max_valid_reward = max(max_valid_reward, result['reward'])

                        if args.wandb:
                            ## record F1, EM for both feedback and TyDi/SQuAD data
                            logger.info('log valid...')
                            wandb.log(
                                {
                                    '(Valid) F1':
                                    result['f1'],
                                    '(Valid) Exact':
                                    result['exact'],
                                    '(Valid) Has Ans F1':
                                    has_ans_eval['f1'],
                                    '(Valid) Has Ans Exact':
                                    has_ans_eval['exact'],
                                    '(Valid) No Ans F1': 
                                    no_ans_eval['f1'],
                                    '(Valid) No Ans Exact': 
                                    no_ans_eval['exact'],
                                    '(Valid) Max F1':
                                    max_valid_f1,
                                    '(Valid) Max Exact':
                                    max_valid_em,
                                    '(Valid) Max Reward':
                                    max_valid_reward,
                                    '(Valid) Reward':
                                    result['reward'],
                                    '(Valid) perc. UNANS':
                                    result['perc. UNANS'],
                                    'perc. UNANS in ANS subset':
                                    result['perc. UNANS in ANS subset'],
                                    'perc. UNANS in UNANS subset':
                                    result['perc. UNANS in UNANS subset'],
                                    'F1 in predicted ANS subset':
                                    result['F1 in predicted ANS subset'],
                                    'F1 in predicted UNANS subset':
                                    result['F1 in predicted UNANS subset'],
                                    'classification_acc':
                                    result['classification_acc'],
                                },
                                step=global_step)
                            if args.valid_data_type == 'feedback':  # validation on feedback
                                wandb.log({
                                    '(Valid) Reward': result['reward'],
                                }, step=global_step)
                    else:
                        save_model = True
                    ######## validation ########

                    if args.save_every:
                        save_model = True

                    if global_step == 0:
                        save_model = False

                    #### model saving ####
                    if save_model and (not args.not_save):
                        logger.info('=====Saving!!!!=====')
                        # save the config; handle multi-gpu
                        if n_gpu > 1:
                            model.module.bert.config.to_json_file(
                                os.path.join(args.output_dir, 'config.json'))
                        else:
                            model.bert.config.to_json_file(
                                os.path.join(args.output_dir, 'config.json'))
                        # save the model
                        ckpt_name = 'saved_checkpoint_%d' % epoch if args.save_every else 'saved_checkpoint'
                        torch.save(
                            {
                                'global_step':
                                global_step,
                                'args':
                                vars(args),
                                'model_state_dict':
                                model.module.state_dict()
                                if n_gpu > 1 else model.state_dict(),  # handle multi-gpu
                                'optimizer_state_dict':
                                optimizer.state_dict(),
                            },
                            os.path.join(args.output_dir, ckpt_name))

                        if best_result:
                            # i.e. best_result is not None
                            filename = EVAL_FILE
                            with open(os.path.join(args.output_dir, filename), "w") as writer:
                                for key in sorted(best_result.keys()):
                                    writer.write("%s = %s\n" % (key, str(best_result[key])))
                                if epoch == 0 and args.eval_metric == 'f1':
                                    one_epoch_f1 = best_result['f1']
                                writer.write("%s = %s\n" % ('one_epoch_f1', one_epoch_f1))

                    #### model saving ####

            ## training reward
            logger.info('(Train) Weighted Reward Per Epoch = %f' %
                        (sum(rewards_per_epoch) / len(rewards_per_epoch)))
            if args.add_classifier:
                logger.info('(Train) Weighted Class Reward Per Epoch = %f' %
                            (sum(class_rewards_per_epoch) / len(class_rewards_per_epoch)))
            if args.wandb:
                wandb.log(
                    {
                        '(Train) Weighted Reward Per Epoch':
                        sum(rewards_per_epoch) / len(rewards_per_epoch),
                    },
                    step=global_step)
            print('ACC PER EPOCH:', acc_per_epoch)


    # that's for testing
    if args.do_eval:
        if args.eval_test:
            table = PrettyTable()
            plot_writer = open(os.path.join(args.output_dir, PLOT_CSV_FILE), "w")


            csv_writer = open(os.path.join(args.output_dir, CSV_FILE), "w")
            csv_writer.write('\t')
            for _ in range(2):
                csv_writer.write(
                    "F1 \t has ans F1 \t no ans F1 \t  EM \t reward \t % unans \t F1 in predicted ANS subset \t"
                )
            for _ in range(2):
                csv_writer.write(
                    "F1 \t has ans F1 \t no ans F1 \t  EM \t % unans \t F1 in predicted ANS subset \t"
                )
            csv_writer.write("F1 \t EM \n")
            csv_writer.write('round %d\t' % args.round_index)
            # look at data/test_files.txt
            # should be [data_type]\t[data_path]
            test_data_list = [tuple(l.strip('\n').split('\t')) for l in open(args.test_file)]
            for test_data_type, test_data_file in test_data_list:
                eval_dataset, eval_examples, eval_features, eval_dataloader, _ = prepare_data(
                    args,
                    test_data_file,
                    tokenizer,
                    data_type=test_data_type,
                    batch_size=args.eval_batch_size,
                    data_split='test')

                # NOTE old: model = BertForQuestionAnsweringSequence.from_pretrained(args.output_dir)
                # model = BertForQuestionAnsweringSequence(model_type=args.model)
                # NOTE change: only evaluate on the test set
                if not args.do_train:
                    if args.model == "deepset/deberta-v3-base-squad2":
                        model = DebertaSQuAD2(model_type=args.model)
                        print('loading deepset model')
                        if args.initialize_model_from_checkpoint:
                            load_initialization(model=model, args=args)
                    else:
                        if args.add_classifier:
                            model = BertForQuestionAnsweringSequence(model_type=args.model)
                        else:
                            model = BertForQuestionAnswering(model_type=args.model)
                        load_initialization(model=model, args=args)
                


                model.to(device)

                logger.info('output_dir: %s' % args.output_dir)
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
                             na_prob_thresh=na_prob_thresh,
                             tokenizer=tokenizer,
                             dataset_name=test_data_type,
                             calculate_score=not args.not_calculate_score,
                             classifier=args.add_classifier
                             )
                with open(
                        os.path.join(
                            args.output_dir,
                            PRED_FILE.split('.')[0] + '-%s.' % test_data_type +
                            PRED_FILE.split('.')[1]), "w") as writer:
                    writer.write(json.dumps(preds, indent=4) + "\n")

                if not args.not_calculate_score:
                    with open(os.path.join(args.output_dir, TEST_FILE), "w") as writer:
                        for key in sorted(result.keys()):
                            writer.write("%s = %s\n" % (key, str(result[key])))

                    table.add_column("[%s] %s" % (test_data_type, "F1"), ['%2.2f' % result['f1']])
                    table.add_column("[%s] %s" % (test_data_type, "EM"),
                                     ['%2.2f' % result['exact']])

                    print(test_data_type)
                    print(result)
                    if 'NoAns_f1' in result:
                        csv_writer.write("%2.2f\t%2.2f\t%2.2f\t%2.2f\t" %
                                         (result['f1'], result['HasAns_f1'], result['NoAns_f1'],
                                          result['exact']))
                    else:
                        assert  test_data_type == 'squad', 'only squad should be without NoAns_f1!'
                        csv_writer.write("%2.2f\t%2.2f\t" % (result['f1'], result['exact']))

                    if 'reward' in result:
                        csv_writer.write("%2.2f\t" % (result['reward']))
                        table.add_column("[%s] %s" % (test_data_type, "Reward"),
                                         ['%2.2f' % result['reward']])
                    
                    if 'classification_acc' in result:
                        csv_writer.write("%2.2f\t" % (result['classification_acc']))
                    if 'perc. UNANS' in result:
                        csv_writer.write("%2.2f\t" % (100*result['perc. UNANS']))

                    if 'F1 in predicted ANS subset' in result:
                        csv_writer.write("%2.2f\t" % (100*result['F1 in predicted ANS subset']))

                    if test_data_type == 'feedback' or test_data_type == 'tydi' or test_data_type == 'squad2':
                        plot_writer.write(test_data_file + '\t')
                        plot_writer.write("%2.2f\t%2.2f\t%2.2f\t"%(result['f1'], result['HasAns_f1'], result['NoAns_f1']))
                        plot_writer.write("%2.2f\t%2.2f\t%2.2f\t"%(100*result['F1 in predicted ANS subset'], 100*result['F1 in predicted UNANS subset'], result['classification_acc']))
                        plot_writer.write("%2.2f\t%2.2f\t%2.2f\t"%(100*result['perc. UNANS'], 100*result['perc. UNANS in ANS subset'], 100*result['perc. UNANS in UNANS subset']))
                        plot_writer.write('\n')
                    # Round   F1  Ans F1  Unans F1    Predicted Ans F1    Predicted Unans F1  CLS Acc %unans  %unans|an   %unans|un
            print(table)
            csv_writer.write('\n')
            csv_writer.close()


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
                        default='data/train-460.jsonl.gz',
                        type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--initial_train_file",
                        default='data/tydiqa-v1.0-train-90%.jsonl.gz',
                        type=str,
                        help="Initial TyDi File for training")
    parser.add_argument("--dev_file",
                        default='data/tydiqa-v1.0-train-10%.jsonl.gz',
                        type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--eval_per_epoch",
                        default=10,
                        type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument(
        "--max_seq_length",
        default=512,
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
                        default=30,
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
    parser.add_argument("--turn_off_dropout", action='store_true', help="Whether turn off dropout")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--scheduler', default='linear', type=str, help='Learning rate scheduler.')
    parser.add_argument('--initialize_model_from_checkpoint',
                        default=None,
                        help='Relative filepath to a saved checkpoint as model initialization.')

    parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')

    #### for bandit learning ####
    parser.add_argument('--wandb', action='store_true', help='Whether to use wandb for logging.')
    parser.add_argument('--notes', default='', help='Notes for this experiment: wandb logging')
    parser.add_argument(
        '--reward_fn',
        default='binary_reward',
        type=str,
        choices=['binary_reward'],
        help='the type of reward function used during training: stick with binary in this work')
    parser.add_argument('--negative_reward',
                        default=-0.1,
                        type=float,
                        help='value for negative update')
    parser.add_argument('--partial_reward',
                        default=0.5,
                        type=float,
                        help='value for negative update')
    parser.add_argument('--reward_wrong_unans',
                        default=-2,
                        type=float,
                        help='value for negative update')
    parser.add_argument('--reward_correct_span',
                        default=2,
                        type=float,
                        help='value for negative update')
    parser.add_argument('--reward_correct_unans',
                        default=0.4,
                        type=float,
                        help='value for negative update')
    parser.add_argument('--reward_class_wrong',
                        default=-1,
                        type=float,
                        help='value for wrong classification prediction')
    parser.add_argument('--reward_class_correct_ans',
                        default=1,
                        type=float,
                        help='value for correct classification prediction ans')
    parser.add_argument('--setup',
                        default='offline',
                        type=str,
                        choices=['offline'],
                        help='offline setup')

    # below are added
    parser.add_argument('--na_prob_thresh',
                        type=float,
                        default=0.0,
                        help='0.0 means no threshholding for na probs')

    parser.add_argument('--test_data_type', type=str, help='which data type to test on')
    parser.add_argument(
        '--prepend_title',
        action='store_true',
        help='whether to prepend the document title to the document text before tokenization')
    parser.add_argument('--feedback_as_valid',
                        action='store_true',
                        help='whether use feedback data as validation data')
    parser.add_argument('--valid_data_type', type=str, help='which data type to valid on')
    parser.add_argument('--tag', type=str, default='', help='wandb tag for experiments')
    parser.add_argument('--checkpoint_name', type=str, default='saved_checkpoint')
    parser.add_argument('--not_save',
                        action='store_true',
                        help='whether not save when doing validation')
    parser.add_argument('--save_every',
                        action='store_true',
                        help='whether save checkpoint every time do validation')
    parser.add_argument('--load_log_prob',
                        action='store_true',
                        help='whether directly load log probability from data')
    parser.add_argument('--rehearsal', action='store_true', help='whether use rehearsal')
    parser.add_argument('--round_index', type=int, default=0)
    parser.add_argument('--not_calculate_score', action='store_true')
    parser.add_argument('--add_classifier', action='store_true', help='whether to add a separate classifier for unans/ans')
    parser.add_argument('--entropy_coeff', type=float, default=0.0)
    parser.add_argument('--class_coeff', type=float, default=1.0)

    args = parser.parse_args()

    main(args)
