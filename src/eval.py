from __future__ import absolute_import, division, print_function
import collections
import json
import logging
import os
import re
import string
import torch
import math

from IPython import embed

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits", "answerable"])


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
                     version_2_with_negative):
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
        result = unique_id_to_result[features[0].unique_id]
        if result.answerable != None and not result.answerable:
            all_predictions[example.qas_id] = ''
            continue
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

                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    if start_index == 0 and end_index != 0:
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
            if result.answerable == None or (not result.answerable):
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
                                                  ["text", "start_logit", "end_logit"])
        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break

            feature = features[pred.feature_index]
            if pred.start_index > 0:
                # this should work!
                orig_doc_start_char = feature.token_to_orig_map[pred.start_index -
                                                                feature.doc_token_offset][0]

                # getting the end index of the answer span (in character)
                if pred.end_index - feature.doc_token_offset >= len(feature.token_to_orig_map):
                    # if the end index is in the [PAD] area, then make it the last token in context
                    orig_doc_end_char = feature.token_to_orig_map[-1][1]
                else:
                    orig_doc_end_char = feature.token_to_orig_map[pred.end_index -
                                                                  feature.doc_token_offset][1]

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
                                 end_logit=pred.end_logit))

        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(text="",
                                     start_logit=null_start_logit,
                                     end_logit=null_end_logit))
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        all_predictions[example.qas_id] = nbest[0].text
    return all_predictions


def make_qid_to_has_ans(dataset, dataset_name='tydi', pred=None):
    qid_to_has_ans = {}
    for entry in dataset:
        if dataset_name == 'tydi':
            # NOTE changed tydi eval
            is_impossible_list = []
            for idx in range(len(entry['annotations'])):
                is_impossible_list.append(
                    entry['annotations'][idx]['minimal_answer']['plaintext_start_byte'] == -1)
            is_impossible = (is_impossible_list.count(True) > is_impossible_list.count(False))
            qid_to_has_ans[entry['example_id']] = not is_impossible
            # qid_to_has_ans[entry['example_id']] = sum([int(entry['annotations'][i]['minimal_answer']['plaintext_start_byte'] == -1) for i in range(len(entry['annotations']))]) == 0
        elif dataset_name == 'squad' or dataset_name == 'nq' or dataset_name == 'newsqa' or dataset_name == 'searchqa':
            qid_to_has_ans[entry['example_id']] = True
        elif dataset_name == 'squad2':
            qid_to_has_ans[entry['example_id']] = not entry['is_impossible']
        elif dataset_name == 'feedback':
            is_impossible_list = []
            for idx in range(len(entry['annotations'])):
                is_impossible_list.append(entry['annotations'][idx]['orig_answer_text'] == '')
            is_impossible = (is_impossible_list.count(True) > is_impossible_list.count(False))
            qid_to_has_ans[entry['example_id']] = not is_impossible

        else:
            raise NotImplementedError
    return qid_to_has_ans


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


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(dataset, preds, partial_reward, negative_reward, calculate_reward=False):
    exact_scores = {}
    f1_scores = {}
    rewards = {}
    classification_accs = {}
    for entry in dataset:
        qid = entry['example_id']
        gold_answers = []
        for a in entry['annotations']:
            if 'orig_answer_text' in a:
                # if the example belongs to SQuAD/SQuAD2.0/feedback
                if normalize_answer(a['orig_answer_text']):
                    gold_answers.append(a['orig_answer_text'])
                else:
                    gold_answers.append('')
            else:
                # if the example belongs to TyDiQA
                plaintext_start_byte = a['minimal_answer']['plaintext_start_byte']
                plaintext_end_byte = a['minimal_answer']['plaintext_end_byte']
                paragraph_text = entry['document_plaintext']
                orig_answer_text = paragraph_text[plaintext_start_byte:plaintext_end_byte]
                if normalize_answer(orig_answer_text):
                    gold_answers.append(orig_answer_text)
                else:
                    gold_answers.append('')

        # changed tydi eval (and for feedback also)
        # deal with majority vote of unans
        if gold_answers.count('') > (len(gold_answers) - gold_answers.count('')):
            gold_answers = ['']
        else:
            span_answers = []
            for span in gold_answers:
                if span != '':
                    span_answers.append(span)
            gold_answers = span_answers

        # for SQuAD 2.0 unanswerable
        if len(gold_answers) == 0:
            gold_answers = ['']

        if qid not in preds:
            print('Missing prediction for %s' % qid)
            continue
        a_pred = preds[qid]

        if gold_answers[0] == '':
            classification_accs[qid] = (a_pred == '')
        else:
            classification_accs[qid] = (a_pred != '')

        exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
        f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
        inst_rewards = []
        if calculate_reward:
            for a in gold_answers:
                f1_score = compute_f1(a, a_pred)
                if a_pred == '':
                    if f1_score == 1:
                        inst_rewards.append(0.4)
                    else:
                        inst_rewards.append(-2)
                else:
                    if f1_score == 1:
                        inst_rewards.append(2)
                    elif f1_score == 0:
                        inst_rewards.append(-0.1)
                    else:
                        inst_rewards.append(1)

            rewards[qid] = max(inst_rewards)
    return exact_scores, f1_scores, rewards, classification_accs


def make_eval_dict(exact_scores=None, f1_scores=None, rewards=None, classification_accs=None, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        out = [('total', total)]
        if exact_scores != None:
            out.append(('exact', 100.0 * sum(exact_scores.values()) / total))
        if f1_scores != None:
            out.append(('f1', 100.0 * sum(f1_scores.values()) / total))
        if rewards != None:
            out.append(('reward', sum(rewards.values())))
        if classification_accs != None:
            out.append(('classification_acc', 100.0 * sum(classification_accs.values()) / total))
        return collections.OrderedDict(out)
    else:
        total = len(qid_list)
        out = [('total', total)]
        if exact_scores != None:
            out.append(('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total))
        if f1_scores != None:
            out.append(('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total))
        if rewards != None:
            out.append(('reward', sum(rewards[k] for k in qid_list)))
        if classification_accs != None:
            out.append(('classification_acc', 100.0 * sum(classification_accs.values()) / total))
        return collections.OrderedDict(out)


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]


def evaluate(args,
             model,
             device,
             eval_dataset,
             eval_dataloader,
             eval_examples,
             eval_features,
             na_prob_thresh,
             tokenizer,
             dataset_name,
             calculate_score=True,
             classifier=False):
    all_results = []
    model.eval()

    for idx, (input_ids, input_mask, segment_ids, example_indices) in enumerate(eval_dataloader):
        if idx % 100 == 0:
            logger.info("Running test: %d / %d" % (idx, len(eval_dataloader)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            if classifier:
                batch_start_logits, batch_end_logits, answerable_logits = model([input_ids, input_mask, segment_ids], classifier=True)
                pred_answerable = answerable_logits.argmax(dim=-1)
            else:
                batch_start_logits, batch_end_logits, _ = model([input_ids, input_mask, segment_ids], classifier=False)

        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            whether_answerable = pred_answerable[i] if classifier else None
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(
                RawResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits, answerable=whether_answerable))

    ### predict for all data points
    preds = make_predictions(eval_examples, eval_features, all_results, args.n_best_size,
                             args.max_answer_length, args.version_2_with_negative)


    if calculate_score:
        ## calculate scores
        has_ans_eval = None
        no_ans_eval = None
        if args.version_2_with_negative:
            # decide which data to test on
            logger.info('validation on: %s' % dataset_name)

            if dataset_name == 'feedback':
                qid_to_has_ans = make_qid_to_has_ans(eval_dataset, dataset_name, pred=preds)
                # split the question ids into answerable/unanswerable
                has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
                no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
                # get the EM/F1/reward scores
                exacts, f1s, rewards, classification_accs = get_raw_scores(eval_dataset,
                                                      preds,
                                                      args.partial_reward,
                                                      args.negative_reward,
                                                      calculate_reward=True)

                stats = {'Correct': 0, 'Partially Correct': 0, 'Wrong': 0, 'Total': 0, 'Unans_Correct':0, 'Ans_Correct':0, 'Unans_Wrong':0, 'Ans_Wrong':0, 'Unans_Partially Correct':0, 'Ans_Partially Correct':0, 'Unans_Total':0, 'Ans_Total':0,}
                for qid, s in exacts.items():
                    stats['Total'] += 1
                    if s == 1:
                        stats['Correct'] += 1
                    elif f1s[qid] > 0.1:
                        stats['Partially Correct'] += 1
                    else:
                        stats['Wrong'] += 1

                for qid, s in exacts.items():
                    if preds[qid] == '':
                        stats['Unans_Total'] += 1
                        if s == 1:
                            stats['Unans_Correct'] += 1
                        elif f1s[qid] > 0.1:
                            stats['Unans_Partially Correct'] += 1
                        else:
                            stats['Unans_Wrong'] += 1
                    else:
                        stats['Ans_Total'] += 1
                        if s == 1:
                            stats['Ans_Correct'] += 1
                        elif f1s[qid] > 0.1:
                            stats['Ans_Partially Correct'] += 1
                        else:
                            stats['Ans_Wrong'] += 1

                logger.info('====================')
                logger.info(
                    'Feedback Type [All]: Total: %d | Correct: %2.2f | Partially Correct: %2.2f | Wrong: %2.2f' %
                    (stats['Total'],
                     (100 * stats['Correct'] / float(stats['Total'])),
                     (100 * stats['Partially Correct'] / float(stats['Total'])),
                     (100 * stats['Wrong'] / float(stats['Total'])))
                    )

                logger.info('Feedback Type : Correct: %d | Partially Correct: %d | Wrong: %d' %
                            ((stats['Correct']), (stats['Partially Correct']), (stats['Wrong'])))
                logger.info('====================')


                result = make_eval_dict(exact_scores=exacts, f1_scores=f1s, rewards=rewards, classification_accs=classification_accs)
                # merge the eval results from all subsets into a single dict
                if has_ans_qids:
                    has_ans_eval = make_eval_dict(exact_scores=exacts,
                                                  f1_scores=f1s,
                                                  rewards=rewards,
                                                  classification_accs=classification_accs,
                                                  qid_list=has_ans_qids)
                    merge_eval(result, has_ans_eval, 'HasAns')
                if no_ans_qids:
                    no_ans_eval = make_eval_dict(exact_scores=exacts,
                                                 f1_scores=f1s,
                                                 rewards=rewards,
                                                 classification_accs=classification_accs,
                                                 qid_list=no_ans_qids)
                    merge_eval(result, no_ans_eval, 'NoAns')
            else:
                # 'tydi'
                qid_to_has_ans = make_qid_to_has_ans(eval_dataset, dataset_name)
                has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
                no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
                exacts, f1s, _, classification_accs = get_raw_scores(eval_dataset, preds, 0, 0, calculate_reward=False)

                result = make_eval_dict(exact_scores=exacts, f1_scores=f1s, classification_accs=classification_accs)
                if has_ans_qids:
                    has_ans_eval = make_eval_dict(exact_scores=exacts,
                                                  f1_scores=f1s,
                                                  classification_accs=classification_accs,
                                                  qid_list=has_ans_qids)
                    merge_eval(result, has_ans_eval, 'HasAns')
                if no_ans_qids:
                    no_ans_eval = make_eval_dict(exact_scores=exacts,
                                                 f1_scores=f1s,
                                                 classification_accs=classification_accs,
                                                 qid_list=no_ans_qids)
                    merge_eval(result, no_ans_eval, 'NoAns')

            # log UNANS ratio: todo add this to the feedback data part?
            result['perc. UNANS'] = sum([int(v == '') for k, v in preds.items()]) * 1.0 / len(preds)
            if has_ans_qids:
                result['perc. UNANS in ANS subset'] = sum(
                    [int(preds[k] == '') for k in has_ans_qids]) * 1.0 / len(has_ans_qids)
            if no_ans_qids:
                result['perc. UNANS in UNANS subset'] = sum(
                    [int(preds[k] == '') for k in no_ans_qids]) * 1.0 / len(no_ans_qids)
            # log precision
            ans_f1, unans_f1, ans_count = 0.0, 0.0, 0
            for k, v in preds.items():
                if v != '':
                    ans_f1 += f1s[k]
                    ans_count += 1
                else:
                    unans_f1 += f1s[k]
            if ans_count:
                result['F1 in predicted ANS subset'] = ans_f1 / ans_count
            else:
                result['F1 in predicted ANS subset'] = 0
            if (len(preds) - ans_count) == 0:
                result['F1 in predicted UNANS subset'] = 0
            else:
                result['F1 in predicted UNANS subset'] = unans_f1 / (len(preds) - ans_count)
        else:
            exact_raw, f1_raw, _, _ = get_raw_scores(eval_dataset, preds, 0, 0, calculate_reward=False)
            result = make_eval_dict(exact_raw, f1_raw)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        return result, has_ans_eval, no_ans_eval, preds
    else:
        return None, None, None, preds
