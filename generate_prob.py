import json
import csv
import string
import re
import gzip
import collections
import math
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from transformers import BertTokenizer,DebertaTokenizerFast, DebertaV2TokenizerFast, AutoTokenizer

from src.data import read_feedback_examples_and_features, get_feedback_data 
from model import BertForQuestionAnswering, DebertaSQuAD2, BertForQuestionAnsweringSequence
from src_utils.merge_data import merge
from src.eval import RawResult, normalize_answer
import argparse


def get_batch_log_prob(start_probs, end_probs, start_samples, end_samples):
    bs = start_samples.shape[0]
    ignored_index = start_probs.size(1)
    start_samples.clamp_(0, ignored_index)
    end_samples.clamp_(0, ignored_index)
    log_prob = start_probs[torch.arange(bs), start_samples].log() + end_probs[torch.arange(bs),
                                                                          end_samples].log()
    return log_prob


def load_initialization(model, ckpt_name):
    ckpt = torch.load(ckpt_name)

    model.load_state_dict(ckpt['model_state_dict'])
    print("Loaded the model state from a saved checkpoint {}".format(ckpt_name))
    return model

def main(train_batches, model, device, add_classifier):
    total = 0
    log_probs = []
    class_log_probs = []
    for step, batch in enumerate(train_batches):
        batch = tuple(t.to(device) for t in batch)

        input_ids, input_mask, segment_ids, start_samples, end_samples, rewards = batch
        with torch.no_grad():
            start_probs, end_probs, class_prob = model(batch=batch[:3], return_prob=True, classifier=add_classifier)
            if args.add_classifier:
                class_log_prob = class_prob.log()

                class_sample = class_log_prob.argmax(dim=-1).item()

        log_prob = get_batch_log_prob(
            start_probs, end_probs, start_samples, end_samples)

        log_probs.append(log_prob)
        if args.add_classifier:
            class_log_probs.append(class_log_prob[:, class_sample])
        total += input_ids.size(0)

    print('='*50)
    print('[logging] Total: %d'%(total))
    print('='*50)

    if add_classifier:
        return torch.cat(log_probs, dim=0), torch.cat(class_log_probs, dim=0)
    else:
        return torch.cat(log_probs, dim=0), None


    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='deepset/deberta-v3-base-squad2', type=str)
    parser.add_argument("--data_file", default=None, type=str, required=True, help='data you wish to generate prob from')
    parser.add_argument("--checkpoint", default=None, type=str, required=True)
    parser.add_argument("--add_classifier", action='store_true')
    parser.add_argument(
        "--outfile",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.")
    args = parser.parse_args()


    #### initialization ####
    model_type = args.model
    data_file = args.data_file
    outfile = args.outfile
    checkpoint = args.checkpoint
    add_classifier = args.add_classifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1


    # tokenization and dataset
    if model_type == 'deepset/deberta-v3-base-squad2':
        tokenizer = AutoTokenizer.from_pretrained(model_type, return_offsets_mapping=True)
    elif 'v3' in model_type:
        tokenizer = DebertaV2TokenizerFast.from_pretrained(model_type, return_offsets_mapping=True)
    else:
        tokenizer = DebertaTokenizerFast.from_pretrained(model_type, return_offsets_mapping=True)
    
    train_dataset = get_feedback_data(data_file)  # original train data

    # load model
    if model_type == "deepset/deberta-v3-base-squad2":
        model = DebertaSQuAD2(model_type=model_type)
    elif model_type == 'microsoft/deberta-v3-base':
        if args.add_classifier:
            model = BertForQuestionAnsweringSequence(model_type=model_type)
        else:
            model = BertForQuestionAnswering(model_type=model_type)
    if checkpoint:
        model = load_initialization(model, checkpoint)
    model = model.to(device)

    # processing examples
    train_examples, train_features = read_feedback_examples_and_features(input_data=train_dataset,
                                                                        negative_reward=-0.1,
                                                                        partial_reward=0.5,
                                                                        reward_wrong_unans=-1,
                                                                        reward_correct_span=1,
                                                                        reward_correct_unans=1,
                                                                        reward_class_wrong=0,
                                                                        reward_class_correct_ans=1,
                                                                        tokenizer=tokenizer,
                                                                        max_seq_length=512,
                                                                        prepend_title=True,
                                                                        load_log_prob=False)

    # read_feedback_examples_and_features(train_dataset, -0.1, 0.5, -1, 1, 1, tokenizer, 512, True)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    
    all_start_samples = torch.tensor([f.start_sample for f in train_features], dtype=torch.long)
    all_end_samples = torch.tensor([f.end_sample for f in train_features], dtype=torch.long)
    all_rewards = torch.tensor([f.reward for f in train_features], dtype=torch.float)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_samples, all_end_samples, all_rewards)
    print("***** Train *****")
    print("  Num examples = %d"%len(train_features))
    print("  Batch size = %d"%batch_size)

    train_dataloader = DataLoader(data, batch_size=batch_size)
    train_batches = [batch for batch in train_dataloader]


    # main function
    log_probs, class_log_probs = main(train_batches, model, device, add_classifier)
    print(log_probs.size(), len(train_dataset))
    assert log_probs.size(0) == len(train_dataset)
    if args.add_classifier:
        assert class_log_probs.size(0) == len(train_dataset)
        print(class_log_probs.size())

    for i, inst in enumerate(train_dataset):
        inst['log_prob'] = log_probs[i].item()
        # print(class_log_probs[i])
        if args.add_classifier:
            inst['class_log_prob'] = class_log_probs[i].item()
        else:
            inst['class_log_prob'] = 0

    print(train_dataset[0])

    # write data
    fw = open(outfile, 'w')
    for l in train_dataset:
        fw.write(json.dumps(l) + '\n')
    fw.close()




