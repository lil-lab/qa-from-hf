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
import csv

from IPython import embed

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from transformers import BertTokenizer, DebertaTokenizer, DebertaTokenizerFast
from transformers import AdamW
from model import BertForQuestionAnswering
from transformers import get_scheduler
from datasets import load_dataset

from pytorch_pretrained_bert.tokenization import BasicTokenizer  # used in evaluation
from pytorch_pretrained_bert.tokenization import whitespace_tokenize


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def make_predictions(all_examples, all_features, all_results, n_best_size, max_answer_length,
                    do_lower_case , verbose_logging, version_2_with_negative):
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result
    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []
        score_null = 1000000
        min_null_feature_index = 0
        null_start_logit = 0
        null_end_logit = 0
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    # if start_index not in feature.token_to_orig_map:
                    #     continue
                    # if end_index not in feature.token_to_orig_map:
                    #     continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(feature_index=feature_index,
                                          start_index=start_index,
                                          end_index=end_index,
                                          start_logit=result.start_logits[start_index],
                                          end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(feature_index=min_null_feature_index,
                                  start_index=0,
                                  end_index=0,
                                  start_logit=null_start_logit,
                                  end_logit=null_end_logit))
        prelim_predictions = sorted(prelim_predictions,
                                    key=lambda x: (x.start_logit + x.end_logit),
                                    reverse=True)


        _NbestPrediction = collections.namedtuple("NbestPrediction",
                                                  ["text", "start_logit", "end_logit", "start_index", "end_index"])
        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break

            feature = features[pred.feature_index]
            if pred.start_index > 0:
                try:
                    orig_doc_start_char = feature.token_to_orig_map[pred.start_index-feature.doc_token_offset][0]
                except IndexError:
                    print('start index', pred.start_index)
                    print('doc_token_offset', feature.doc_token_offset)
                    print('feature map', feature.token_to_orig_map)
                    print('len feature map', len(feature.token_to_orig_map))
                    print('token_is_max_context', feature.token_is_max_context)
                    exit(0)

                # getting the end index of the answer span (in character)
                if pred.end_index-feature.doc_token_offset >= len(feature.token_to_orig_map):
                    # if the end index is in the [PAD] area, then make it the last token in context
                    orig_doc_end_char = feature.token_to_orig_map[-1][1]
                else:
                    orig_doc_end_char = feature.token_to_orig_map[pred.end_index-feature.doc_token_offset][1]

                ans_text = example.paragraph_text[orig_doc_start_char:orig_doc_end_char]

                ans_text = ans_text.replace(" ##", "")
                ans_text = ans_text.replace("##", "")
                ans_text = ans_text.strip()
                final_text = ans_text

                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(text=final_text,
                                 start_logit=pred.start_logit,
                                 end_logit=pred.end_logit,
                                 start_index=pred.start_index,
                                 end_index=pred.end_index))

        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(text="",
                                     start_logit=null_start_logit,
                                     end_logit=null_end_logit,
                                     start_index=pred.start_index,
                                     end_index=pred.end_index))
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_index=pred.start_index, end_index=pred.end_index))

        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_index=pred.start_index, end_index=pred.end_index))
        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)
        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            # all_predictions[example.qas_id] = best_non_null_entry.text
            all_predictions[example.qas_id] = (best_non_null_entry.text, best_non_null_entry.start_index, best_non_null_entry.end_index)
        all_nbest_json[example.qas_id] = nbest_json

    return all_predictions, all_nbest_json, scores_diff_json
