import collections
import json
import logging
import os
import random
import time
from io import open
import gzip
import datetime
import csv
import string
import re

from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

INITIAL_REWARD = 1


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """
    def __init__(self,
                 qas_id,
                 question_text,
                 paragraph_text,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.paragraph_text = paragraph_text
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (self.question_text)
        s += ", paragraph_text: %s" % (self.paragraph_text)
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class FeedbackFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 token_to_orig_map,
                 doc_token_offset,
                 input_ids,
                 input_mask,
                 segment_ids,
                 token_is_max_context,
                 start_sample=None,
                 end_sample=None,
                 class_sample=None,
                 reward=0,
                 class_reward=0,
                 log_prob=0,
                 class_log_prob=0):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.doc_token_offset = doc_token_offset
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.token_is_max_context = token_is_max_context
        self.start_sample = start_sample
        self.end_sample = end_sample
        self.class_sample = class_sample
        self.reward = reward
        self.class_reward = class_reward
        self.log_prob = log_prob
        self.class_log_prob = class_log_prob


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


def get_tydi_data(input_file):
    """
    Only load Enligsh data from tydi qa dataset
    """
    input_data = []
    with gzip.GzipFile(input_file, 'r') as reader:
        # skip header
        content = reader.read().decode('utf-8').strip().split('\n')[1:]
        for line in content:
            ex = json.loads(line)
            # only choose english
            if ex['language'] == 'english':
                YES_NO = [annotation['yes_no_answer'] for annotation in ex['annotations']]
                # ignore YES/NO examples
                if 'YES' not in YES_NO:
                    input_data.append(ex)
    assert (len(input_data) != 0)
    return input_data

def get_mrqa_data(input_file):
    input_data = []
    id_=0
    with gzip.GzipFile(input_file, 'r') as reader:
        # skip header
        content = reader.read().decode('utf-8').strip().split('\n')[1:]
        for line in content:
            ex = json.loads(line)
            for qa in ex['qas']:
                inst = {}
                inst['title'] = ''
                inst['context'] = ex['context']
                inst['question'] = qa['question']
                inst['example_id'] = str(id_)
                inst['id'] = str(inst['example_id'])
                inst['annotations'] = [{'orig_answer_text':l} for l in qa['answers']]
                inst['detected_answers'] = qa['detected_answers']
                id_ += 1
                input_data.append(inst)
    assert (len(input_data) != 0)
    return input_data

def get_nq_data(input_file):
    """
    Only load Enligsh data from tydi qa dataset
    """
    input_data = []
    id_ = 0
    with gzip.GzipFile(input_file, 'r') as reader:
        # skip header
        content = reader.read().decode('utf-8').strip().split('\n')[1:]
        for line in content:
            ex = json.loads(line)

            for qa in ex['qas']:
                inst = {}
                inst['title'] = ''
                inst['context'] = ex['context']
                inst['question'] = qa['question']
                inst['example_id'] = str(id_)
                inst['id'] = str(inst['example_id'])
                inst['annotations'] = [{'orig_answer_text': l} for l in qa['answers']]
                id_ += 1
                input_data.append(inst)

    assert (len(input_data) != 0)
    return input_data


def get_feedback_data(input_file):
    # read data
    with gzip.GzipFile(input_file, 'r') as reader:
        content = reader.read().decode('utf-8').strip().split('\n')
        input_data = [json.loads(line) for line in content]

    for i, inst in enumerate(input_data):
        inst['example_id'] = str(i)

        ## for validation, keep the format the same as SQuAD data
        if 'annotations' in inst:
            if 'orig_answer_text' not in inst['annotations'][0]:
                inst['annotations'] = [{'orig_answer_text': l} for l in inst['annotations']]
        elif 'annotation' in inst:
            inst['annotations'] = [{'orig_answer_text': inst['annotation']}]
    return input_data


# https://github.com/google-research-datasets/tydiqa/blob/43cde6d598c1cf88c1a8b9ed32e89263ffb5e03b/tydi_eval.py#L239
# return a string
def byte_slice(text, start, end):
    byte_str = bytes(text, 'utf-8')
    # return str(byte_str[start:end])
    return byte_str[start:end].decode('utf-8')



def read_mrqa_examples_and_features(is_training,
                                     version_2_with_negative,
                                     tokenizer,
                                     max_seq_length,
                                     prepend_title,
                                     input_data,
                                     get_dataset=False):
    """Read a SQuAD json file into a list of SquadExample."""

    assert (len(input_data) != 0)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    dataset = []
    features = []
    truncated_cnt = 0
    filtered = 0
    unique_id = 1000000000
    example_index = 0
    for entry in input_data:
        title = entry["title"]
        paragraph_text = entry["context"]
        qas_id = entry["id"]
        question_text = entry["question"]
        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False
        padding_length = 0
        
        # prepare inputs
        query_tokens = tokenizer.tokenize(question_text)
        context_tokens = tokenizer.tokenize(paragraph_text)
        if prepend_title:
            title_tokens = tokenizer.tokenize(title)

        # truncate if needed
        if prepend_title:
            max_context_length = max_seq_length - len(query_tokens) - len(title_tokens) - 4
        else:
            max_context_length = max_seq_length - len(query_tokens) - 3

        if len(query_tokens) > 100:
            filtered += 1
            continue

        if len(context_tokens) > max_context_length:
            context_tokens = context_tokens[0:max_context_length]
        else:
            padding_length = max_context_length - len(context_tokens)

        # convert to indices
        tokens = []
        segment_ids = []
        token_is_max_context = {}
        doc_token_offset = 0
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        # prepend title tokens
        if prepend_title:
            for token in title_tokens:
                tokens.append(token)
                segment_ids.append(2)
            tokens.append("[SEP]")
            segment_ids.append(2)

        doc_token_offset = len(tokens)
        for token in context_tokens:
            token_is_max_context[len(tokens)] = True
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        if padding_length > 0:
            for _ in range(padding_length):
                tokens.append("[PAD]")
                segment_ids.append(0)


        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * (max_seq_length - padding_length) + [0] * padding_length

        # filter [CLS] token to get the real offset mapping
        offset_mapping = tokenizer(paragraph_text, 
                                   return_offsets_mapping=True, 
                                   truncation=True, 
                                   max_length=max_context_length + 
                                   2)['offset_mapping'][1:-1] 

        # get character to wrod offset for the document(context)
        char_to_word_offset = {}
        for index, offset in enumerate(offset_mapping):  
            assert (offset[0] + offset[1])
            for o in range(offset[0], offset[1]):
                char_to_word_offset[o] = index

        
        orig_answer_text = entry["detected_answers"][0]["text"]
        answer_length = len(orig_answer_text)
        char_start = entry["detected_answers"][0]["char_spans"][0][0]
        char_end = entry["detected_answers"][0]["char_spans"][0][1]
        # char_end = char_start + answer_length - 1

        # check if the answer is truncated by tokenization
        if (char_start not in char_to_word_offset or char_end not in char_to_word_offset):
            max_char = max([k for k in char_to_word_offset.keys()])

            start_position = -1
            end_position = -1
            orig_answer_text = ""

            if is_training:
                truncated_cnt += 1
                continue
        else:
            start_position = char_to_word_offset[char_start] + doc_token_offset
            end_position = char_to_word_offset[char_end] + doc_token_offset

        # check if the tokenized answer matches the original one
        if normalize_answer(orig_answer_text).replace(" ", "") != normalize_answer("".join(
            paragraph_text[offset_mapping[start_position-doc_token_offset][0]:
                offset_mapping[end_position - 
                    doc_token_offset][1]])).replace(" ", ""):
            if is_training:
                filtered += 1
                continue    


        example = SquadExample(qas_id=qas_id,
                       question_text=question_text,
                       paragraph_text=paragraph_text,
                       orig_answer_text=orig_answer_text,
                       start_position=start_position,
                       end_position=end_position,
                       is_impossible=False)
        examples.append(example)

        features.append(
                FeedbackFeatures(unique_id=unique_id,
                     example_index=example_index,
                     tokens=tokens,
                     token_to_orig_map=offset_mapping,
                     token_is_max_context=token_is_max_context,
                     doc_token_offset=doc_token_offset,
                     input_ids=input_ids,
                     input_mask=input_mask,
                     segment_ids=segment_ids,
                     start_sample=start_position,
                     end_sample=end_position,
                     reward=INITIAL_REWARD
                )
            )
        unique_id += 1
        example_index += 1

        if get_dataset:
            dataset.append(entry)

    print('filtered %d examples..., truncated %d examples'%(filtered, truncated_cnt))
    
    assert len(features) == len(examples)
    if get_dataset:
        assert len(dataset) == len(examples)
    return examples, dataset, features

def read_squad_examples_and_features(is_training,
                                     version_2_with_negative,
                                     tokenizer,
                                     max_seq_length,
                                     prepend_title,
                                     input_data,
                                     get_dataset=True):
    """Read a SQuAD json file into a list of SquadExample."""

    assert (len(input_data) != 0)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    dataset = []
    features = []
    truncated_cnt = 0
    filtered = 0
    unique_id = 1000000000
    example_index = 0
    for entry in input_data:
        title = entry["title"]
        paragraph_text = entry["context"]
        qas_id = entry["id"]
        question_text = entry["question"]
        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False
        padding_length = 0

        # prepare inputs
        query_tokens = tokenizer.tokenize(question_text)
        context_tokens = tokenizer.tokenize(paragraph_text)
        if prepend_title:
            title_tokens = tokenizer.tokenize(title)

        # truncate if needed
        if prepend_title:
            max_context_length = max_seq_length - len(query_tokens) - len(title_tokens) - 4
        else:
            max_context_length = max_seq_length - len(query_tokens) - 3

        if len(query_tokens) > 100:
            filtered += 1
            continue

        if len(context_tokens) > max_context_length:
            context_tokens = context_tokens[0:max_context_length]
        else:
            padding_length = max_context_length - len(context_tokens)

        # convert to indices
        tokens = []
        segment_ids = []
        token_is_max_context = {}
        doc_token_offset = 0
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        # prepend title tokens
        if prepend_title:
            for token in title_tokens:
                tokens.append(token)
                segment_ids.append(2)
            tokens.append("[SEP]")
            segment_ids.append(2)

        doc_token_offset = len(tokens)
        for token in context_tokens:
            token_is_max_context[len(tokens)] = True
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        if padding_length > 0:
            for _ in range(padding_length):
                tokens.append("[PAD]")
                segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * (max_seq_length - padding_length) + [0] * padding_length

        # filter [CLS] token to get the real offset mapping
        offset_mapping = tokenizer(paragraph_text,
                                   return_offsets_mapping=True,
                                   truncation=True,
                                   max_length=max_context_length + 2)['offset_mapping'][1:-1]

        # get character to wrod offset for the document(context)
        char_to_word_offset = {}
        for index, offset in enumerate(offset_mapping):
            assert (offset[0] + offset[1])
            for o in range(offset[0], offset[1]):
                char_to_word_offset[o] = index

        # get the start & end positions for the answer
        if is_training:
            if version_2_with_negative:
                is_impossible = (len(entry["answers"]["text"]) == 0)

            if (len(entry["answers"]["text"]) != 1) and (not is_impossible):
                raise ValueError("For training, each question should have exactly 1 answer.")
            if not is_impossible:
                orig_answer_text = entry["answers"]["text"][0]
                answer_length = len(orig_answer_text)
                char_start = entry["answers"]["answer_start"][0]
                char_end = char_start + answer_length - 1

                # check if the answer is truncated by tokenization
                if char_start not in char_to_word_offset or char_end not in char_to_word_offset:
                    truncated_cnt += 1
                    start_position = -1
                    end_position = -1
                    orig_answer_text = ""
                    is_impossible = True
                    continue
                else:
                    start_position = char_to_word_offset[char_start] + doc_token_offset
                    end_position = char_to_word_offset[char_end] + doc_token_offset

                # check if the tokenized answer matches the original one
                if normalize_answer(orig_answer_text).replace(" ", "") != normalize_answer("".join(
                        paragraph_text[offset_mapping[start_position - doc_token_offset][0]:
                                       offset_mapping[end_position -
                                                      doc_token_offset][1]])).replace(" ", ""):
                    filtered += 1
                    continue
            else:
                start_position = -1
                end_position = -1
                orig_answer_text = ""

        example = SquadExample(qas_id=qas_id,
                               question_text=question_text,
                               paragraph_text=paragraph_text,
                               orig_answer_text=orig_answer_text,
                               start_position=start_position,
                               end_position=end_position,
                               is_impossible=version_2_with_negative and is_impossible)
        examples.append(example)
        if get_dataset:
            dataset.append({
                'question_text':
                question_text,
                'example_id':
                qas_id,
                'context':
                paragraph_text,
                # for SQuAD 2.0, 'annotations' would be an empty list
                'annotations': [{
                    'orig_answer_text': a
                } for a in entry['answers']['text']],
                'is_impossible': (len(entry["answers"]["text"]) == 0),
            })

        features.append(
            FeedbackFeatures(unique_id=unique_id,
                             example_index=example_index,
                             tokens=tokens,
                             token_to_orig_map=offset_mapping,
                             token_is_max_context=token_is_max_context,
                             doc_token_offset=doc_token_offset,
                             input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             start_sample=start_position,
                             end_sample=end_position,
                             reward=INITIAL_REWARD))
        unique_id += 1
        example_index += 1

    print('filtered %d examples..., truncated %d examples' % (filtered, truncated_cnt))
    if get_dataset:
        assert len(examples) == len(dataset)
    assert len(features) == len(examples)
    return examples, dataset, features


def read_tydi_examples_and_features(input_data, is_training, version_2_with_negative, tokenizer,
                                    max_seq_length, prepend_title):
    """Read a tydi json file into a list of SquadExample."""
    unique_id = 1000000000

    examples = []
    features = []
    num_impos = 0
    example_index = 0
    truncated_cnt = 0
    for entry in input_data:

        is_impossible_list = []
        for idx in range(len(entry['annotations'])):
            is_impossible_list.append(
                entry['annotations'][idx]['minimal_answer']['plaintext_start_byte'] == -1)
        is_impossible = (is_impossible_list.count(True) > is_impossible_list.count(False))

        if is_training:
            # for training, only 1 annotation
            pas_index = entry['annotations'][0]['passage_answer']['candidate_index']
            if pas_index == -1:
                pas_index = 0
        else:
            # for eval
            if is_impossible:  # if is unanswerable, should be unanswerable for any passage (?)
                pas_index = 0
            else:
                pas_index = -1
                annotation_idx = 0
                while pas_index == -1:
                    pas_index = entry['annotations'][annotation_idx]['passage_answer'][
                        'candidate_index']
                    annotation_idx += 1

        # skip pas_index=1 if only having 1 passage
        if pas_index >= len(entry['passage_answer_candidates']):
            continue

        # get passage and answer starting position in byte
        pas_start_byte = entry['passage_answer_candidates'][pas_index]['plaintext_start_byte']
        pas_end_byte = entry['passage_answer_candidates'][pas_index]['plaintext_end_byte']
        plaintext_start_byte = entry['annotations'][0]['minimal_answer']['plaintext_start_byte']
        plaintext_end_byte = entry['annotations'][0]['minimal_answer']['plaintext_end_byte']
        relative_start_byte = plaintext_start_byte - pas_start_byte
        relative_end_byte = plaintext_end_byte - pas_start_byte

        paragraph_text = byte_slice(entry['document_plaintext'], pas_start_byte, pas_end_byte)

        # get byte to char offset
        byte_to_char = []
        for idx, c in enumerate(paragraph_text):
            for _ in range(len(c.encode())):
                byte_to_char.append(idx)

        qas_id = entry['example_id']  # FIXME shared by two passages
        question_text = entry['question_text']
        title = entry['document_title']
        start_position = None
        end_position = None
        orig_answer_text = byte_slice(text=entry['document_plaintext'],
                                      start=plaintext_start_byte,
                                      end=plaintext_end_byte)

        padding_length = 0

        # prepare inputs
        query_tokens = tokenizer.tokenize(question_text)
        context_tokens = tokenizer.tokenize(paragraph_text)
        if prepend_title:
            title_tokens = tokenizer.tokenize(title)

        # truncate if needed
        if prepend_title:
            max_context_length = max_seq_length - len(query_tokens) - len(title_tokens) - 4
        else:
            max_context_length = max_seq_length - len(query_tokens) - 3

        if len(context_tokens) > max_context_length:
            context_tokens = context_tokens[0:max_context_length]
        else:
            padding_length = max_context_length - len(context_tokens)

        # convert to indices
        tokens = []
        segment_ids = []
        token_is_max_context = {}
        doc_token_offset = 0
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        # prepend title tokens
        if prepend_title:
            for token in title_tokens:
                tokens.append(token)
                segment_ids.append(2)
            tokens.append("[SEP]")
            segment_ids.append(2)

        doc_token_offset = len(tokens)
        for token in context_tokens:
            token_is_max_context[len(tokens)] = True
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        if padding_length > 0:
            for _ in range(padding_length):
                tokens.append("[PAD]")
                segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * (max_seq_length - padding_length) + [0] * padding_length

        # filter [CLS] token to get the real offset mapping
        offset_mapping = tokenizer(paragraph_text,
                                   return_offsets_mapping=True,
                                   truncation=True,
                                   max_length=max_context_length + 2)['offset_mapping'][1:-1]

        # get character to wrod offset for the document(context)
        char_to_word_offset = {}
        for index, offset in enumerate(offset_mapping):
            assert (offset[0] + offset[1])
            for o in range(offset[0], offset[1]):
                char_to_word_offset[o] = index

        if is_training:
            if version_2_with_negative:
                assert is_impossible == (relative_start_byte < 0) or (
                    relative_start_byte >=
                    (pas_end_byte - pas_start_byte)) or (relative_end_byte >=
                                                         (pas_end_byte - pas_start_byte))

            if (len(entry['annotations']) != 1) and (not is_impossible):
                raise ValueError("For training, each question should have exactly 1 answer.")

            if not is_impossible:
                char_start = byte_to_char[relative_start_byte]
                char_end = byte_to_char[relative_end_byte - 1]

                if char_start not in char_to_word_offset:
                    truncated_cnt += 1
                    start_position = -1
                    end_position = -1
                    orig_answer_text = ""
                    is_impossible = True
                else:
                    start_position = char_to_word_offset[char_start] + doc_token_offset
                    end_position = char_to_word_offset[char_end] + doc_token_offset

            else:
                start_position = -1
                end_position = -1
                orig_answer_text = ""
                num_impos += 1

        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            paragraph_text=paragraph_text,
            orig_answer_text=orig_answer_text,
            start_position=(start_position if start_position == None else start_position -
                            doc_token_offset),
            end_position=(end_position if end_position == None else end_position -
                          doc_token_offset),
            is_impossible=is_impossible)
        examples.append(example)

        features.append(
            FeedbackFeatures(unique_id=unique_id,
                             example_index=example_index,
                             tokens=tokens,
                             token_to_orig_map=offset_mapping,
                             token_is_max_context=token_is_max_context,
                             doc_token_offset=doc_token_offset,
                             input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             start_sample=start_position,
                             end_sample=end_position,
                             reward=INITIAL_REWARD))
        unique_id += 1
        example_index += 1

    print('truncated', truncated_cnt)
    assert len(examples) == len(features)
    return examples, features


def read_feedback_examples_and_features(input_data,
                                        negative_reward,
                                        partial_reward,
                                        reward_wrong_unans,
                                        reward_correct_span,
                                        reward_correct_unans,
                                        reward_class_wrong,
                                        reward_class_correct_ans,
                                        tokenizer,
                                        max_seq_length,
                                        prepend_title,
                                        load_log_prob=False):
    unique_id = 1000000000

    examples = []
    features = []

    i = 0
    for inst in input_data:
        qas_id = inst['example_id']
        question_text = inst['question']
        paragraph_text = inst['context']
        feedback = inst['feedback']
        title = inst['topic'] + ' [SEP] ' + inst['aspect']
        startidx = inst['startidx']
        endidx = inst['endidx']

        padding_length = 0

        # prepare inputs
        query_tokens = tokenizer.tokenize(question_text)
        context_tokens = tokenizer.tokenize(paragraph_text)
        if prepend_title:
            title_tokens = tokenizer.tokenize(title)

        # truncate if needed
        if prepend_title:
            max_context_length = max_seq_length - len(query_tokens) - len(title_tokens) - 4
        else:
            max_context_length = max_seq_length - len(query_tokens) - 3

        if len(context_tokens) > max_context_length:
            context_tokens = context_tokens[0:max_context_length]
        else:
            padding_length = max_context_length - len(context_tokens)

        # convert to indices
        tokens = []
        segment_ids = []
        token_is_max_context = {}
        doc_token_offset = 0
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        # prepend title tokens
        if prepend_title:
            for token in title_tokens:
                tokens.append(token)
                segment_ids.append(2)
            tokens.append("[SEP]")
            segment_ids.append(2)

        doc_token_offset = len(tokens)
        for token in context_tokens:
            token_is_max_context[len(tokens)] = True
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        if padding_length > 0:
            for _ in range(padding_length):
                tokens.append("[PAD]")
                segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * (max_seq_length - padding_length) + [0] * padding_length

        offset_mapping = tokenizer(
            paragraph_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_context_length +
            2)['offset_mapping'][1:-1]  # filter [CLS] token to get the real offset mapping

        # get character to wrod offset for the document(context)
        char_to_word_offset = {}
        for index, offset in enumerate(offset_mapping):
            assert (offset[0] + offset[1])
            for o in range(offset[0], offset[1]):
                char_to_word_offset[o] = index

        if feedback in ['Unanswerable', 'Answerable']:
            # just as placeholder, not for eval or training
            start_position = -1
            end_position = -1
            class_position = 0
            reward = 0
            class_reward = 0
        else:
            pred = inst['pred']
            # handle unanswerable
            if pred == '[Unanswerable given the paragraph below]':
                start_position = -1
                end_position = -1
                class_position = 0
            else:
                start_position = startidx
                end_position = endidx
                class_position = 1

            if pred == '[Unanswerable given the paragraph below]': # predict unans, second action reward = 0
                reward = 0
                if feedback == 'Correct':
                    class_reward = 1
                elif feedback == 'Wrong':
                    class_reward = -1
                else:
                    class_reward = 0
            else:
                if feedback == 'Correct':
                    class_reward = reward_class_correct_ans
                    reward = reward_correct_span
                elif feedback == 'Wrong':
                    class_reward = reward_class_wrong
                    reward = negative_reward
                else:
                    class_reward = reward_class_correct_ans
                    reward = partial_reward * reward_correct_span


        if load_log_prob:
            assert 'log_prob' in inst
            log_prob = float(inst['log_prob'])
            assert 'class_log_prob' in inst
            class_log_prob = float(inst['class_log_prob'])
        else:
            log_prob = 0
            class_log_prob = 0

        assert log_prob <= 0
        assert class_log_prob <= 0
        features.append(
            FeedbackFeatures(unique_id=unique_id,
                             example_index=i,
                             tokens=tokens,
                             token_to_orig_map=offset_mapping,
                             token_is_max_context=token_is_max_context,
                             doc_token_offset=doc_token_offset,
                             input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             start_sample=start_position,
                             end_sample=end_position,
                             class_sample=class_position,
                             reward=reward,
                             class_reward=class_reward,
                             log_prob=log_prob,
                             class_log_prob=class_log_prob
                             ))
        examples.append(
            SquadExample(qas_id=qas_id,
                         question_text=question_text,
                         paragraph_text=paragraph_text,
                         start_position=start_position,
                         end_position=end_position))

        # keep track of counts
        i += 1
        unique_id += 1

    return examples, features


def read_feedback_examples_and_features_supervised(input_data,
                                        tokenizer,
                                        max_seq_length,
                                        prepend_title):
    unique_id = 1000000000

    dataset = []
    examples = []
    features = []

    i = 0
    for inst in input_data:
        qas_id = inst['example_id']
        question_text = inst['question']
        paragraph_text = inst['context']
        feedback = inst['feedback']

        if feedback != 'Correct' and feedback not in ['Unanswerable', 'Answerable']:
            continue

        title = inst['topic'] + ' [SEP] ' + inst['aspect']
        startidx = inst['startidx']
        endidx = inst['endidx']

        padding_length = 0

        # prepare inputs
        query_tokens = tokenizer.tokenize(question_text)
        context_tokens = tokenizer.tokenize(paragraph_text)
        if prepend_title:
            title_tokens = tokenizer.tokenize(title)

        # truncate if needed
        if prepend_title:
            max_context_length = max_seq_length - len(query_tokens) - len(title_tokens) - 4
        else:
            max_context_length = max_seq_length - len(query_tokens) - 3

        if len(context_tokens) > max_context_length:
            context_tokens = context_tokens[0:max_context_length]
        else:
            padding_length = max_context_length - len(context_tokens)

        # convert to indices
        tokens = []
        segment_ids = []
        token_is_max_context = {}
        doc_token_offset = 0
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        # prepend title tokens
        if prepend_title:
            for token in title_tokens:
                tokens.append(token)
                segment_ids.append(2)
            tokens.append("[SEP]")
            segment_ids.append(2)

        doc_token_offset = len(tokens)
        for token in context_tokens:
            token_is_max_context[len(tokens)] = True
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        if padding_length > 0:
            for _ in range(padding_length):
                tokens.append("[PAD]")
                segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * (max_seq_length - padding_length) + [0] * padding_length

        offset_mapping = tokenizer(
            paragraph_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_context_length +
            2)['offset_mapping'][1:-1]  # filter [CLS] token to get the real offset mapping

        # get character to wrod offset for the document(context)
        char_to_word_offset = {}
        for index, offset in enumerate(offset_mapping):
            assert (offset[0] + offset[1])
            for o in range(offset[0], offset[1]):
                char_to_word_offset[o] = index

        
        if feedback in ['Unanswerable', 'Answerable']:
            # just as placeholder, not for eval or training
            start_position = -1
            end_position = -1
            reward = 0
        else:
            pred = inst['pred']
            # handle unanswerable
            if pred == '[Unanswerable given the paragraph below]':
                start_position = -1
                end_position = -1
            else:
                start_position = startidx
                end_position = endidx

            
        features.append(
            FeedbackFeatures(unique_id=unique_id,
                             example_index=i,
                             tokens=tokens,
                             token_to_orig_map=offset_mapping,
                             token_is_max_context=token_is_max_context,
                             doc_token_offset=doc_token_offset,
                             input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             start_sample=start_position,
                             end_sample=end_position,
                             reward=INITIAL_REWARD,
                             log_prob=0))
        examples.append(
            SquadExample(qas_id=qas_id,
                         question_text=question_text,
                         paragraph_text=paragraph_text,
                         start_position=start_position,
                         end_position=end_position))

        dataset.append(inst)
        # keep track of counts
        i += 1
        unique_id += 1

    return examples, dataset, features
