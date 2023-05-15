from rehearsal import get_feedback_data
from transformers import BertTokenizer, DebertaTokenizer, DebertaV2TokenizerFast, DebertaTokenizerFast
import re
import json
from tqdm import tqdm
import csv
import collections
import matplotlib.pyplot as plt



def analyze_data(data, round_idx, test=False):
    stats = {'correct': 0, 'wrong':0, 'unans_correct':0, 'ans_correct':0, 'unans_wrong':0, 'ans_wrong':0, 'feedback_total': 0, 'num_unans':0, 'time':0, 'annotation_total':0, 'annotation_answerable':0, 'annotation_unanswerable':0}
    seen_workers = set()
    seen_workers = set()
    class_reward = 0.0
    reward = 0.0
    total_reward = 0.0
    total_ans = 0
    data_per_file = []

    for dict_ in tqdm(data):
        if dict_['workerId'] not in seen_workers:
            seen_workers.add(dict_['workerId'])
        
        if not dict_['question']:
            continue
        if dict_['feedback'] in ["Answerable", "Unanswerable"]:
            stats['annotation_total'] += 1
            if dict_['feedback'] == 'Answerable':
                stats['annotation_answerable'] += 1
            else:
                stats['annotation_unanswerable'] += 1
        else:

            stats['feedback_total'] += 1
            unans = int(dict_['pred'] == '[Unanswerable given the paragraph below]')
            stats['num_unans'] += unans
            total_ans += (1 - unans)

            if dict_['feedback'] == 'Correct': # if correct
                stats['correct'] += 1
                stats['unans_correct'] += unans
                stats['ans_correct'] += (1 - unans)
                class_reward += 1
                if unans != 1:
                    reward += 1
                    total_reward += 2
                else:
                    total_reward += 1
            elif dict_['feedback'] == 'Wrong': # if wrong
                stats['wrong'] += 1
                stats['unans_wrong'] += unans
                stats['ans_wrong'] += (1 - unans)
                if unans == 1:  # if unans
                    class_reward -= 1
                    total_reward -= 1
                else:
                    reward -= 0.1
                    total_reward -= 0.1
            else:
                if unans != 1:  # if ans
                    class_reward += 1
                    reward += 0.5
                    total_reward += 1.5


    stats['correct'] /= float(stats['feedback_total'])
    stats['wrong'] /= float(stats['feedback_total'])
    stats['partial'] = 1 - stats['correct'] - stats['wrong']

    stats['unans_correct'] /= float(stats['num_unans'])
    stats['unans_wrong'] /= float(stats['num_unans'])
    stats['unans_partial'] = 1 - stats['unans_correct'] - stats['unans_wrong']

    stats['ans_correct'] /= float(stats['feedback_total'] - stats['num_unans'])
    stats['ans_wrong'] /= float(stats['feedback_total'] - stats['num_unans'])
    stats['ans_partial'] = 1 - stats['ans_correct'] - stats['ans_wrong']

    print('======= Data Results =======')
    print('Total: %d | Correct Per.: %2.2f | Partially Correct Per.: %2.2f | Wrong Per.: %2.2f'%(stats['feedback_total'], stats['correct'] * 100, stats['partial'] * 100, stats['wrong'] * 100))
    print('UnAns: %d | Correct Per.: %2.2f | Partially Correct Per.: %2.2f | Wrong Per.: %2.2f'%(stats['num_unans'], stats['unans_correct'] * 100, stats['unans_partial'] * 100, stats['unans_wrong'] * 100))
    print('Ans: %d | Correct Per.: %2.2f | Partially Correct Per.: %2.2f | Wrong Per.: %2.2f'%(stats['feedback_total'] - stats['num_unans'], stats['ans_correct'] * 100, stats['ans_partial'] * 100, stats['ans_wrong'] * 100))

    if test:
        print('======= Annotation Results =======')
        print('Total: %d | Answerable: %d | Unanswerable: %d'%(stats['annotation_total'], stats['annotation_answerable'], stats['annotation_unanswerable']))

    print('=============================')
    print('Num Unique Workers: %d'%len(seen_workers))

    csv_data = [['Round %d - Feedback'%round_idx, '', '', '', '', '', '', 'Round %d - Test'%round_idx], ['', '(%) exs', '# exs', 'Correct (%)', 'Partially Correct (%)', 'Wrong (%)', '', '# exs', 'Answerable (%)', 'Unanswerable (%)']]
    csv_data.append(['All', '100', '%d'%stats['feedback_total'], '%2.2f'%(stats['correct'] * 100), '%2.2f'%(stats['partial'] * 100), '%2.2f'%(stats['wrong'] * 100), '', '%d'%stats['annotation_total'], '%2.2f'%(100*stats['annotation_answerable']/float(stats['annotation_total'])), '%2.2f'%(100*stats['annotation_unanswerable']/float(stats['annotation_total']))])
    csv_data.append(['Unanswerable', '%2.2f'%(100*stats['num_unans']/float(stats['feedback_total'])), '%d'%stats['num_unans'], '%2.2f'%(stats['unans_correct'] * 100), '%2.2f'%(stats['unans_partial'] * 100), '%2.2f'%(stats['unans_wrong'] * 100)])
    csv_data.append(['Answerable', '%2.2f'%(100*(stats['feedback_total'] - stats['num_unans'])/float(stats['feedback_total'])), '%d'%(stats['feedback_total'] - stats['num_unans']), '%2.2f'%(stats['ans_correct'] * 100), '%2.2f'%(stats['ans_partial'] * 100), '%2.2f'%(stats['ans_wrong'] * 100)])
    return csv_data

def compute_question_answer_lengths(data, tokenizer, round_idx, pred_list=None, is_training=False):
    # for lengths
    data_list = [data]
    lengths = [[] for _ in range(len(data_list))]
    if not is_training:
        answer_lengths = [{'Correct':[0], 'Partially Correct':[0], 'Wrong':[0], 'All':[]} for _ in range(len(data_list))]
    else:
        answer_lengths = [{'Correct':[], 'Partially Correct':[], 'Wrong':[], 'All':[]} for _ in range(len(data_list))]
    csv_data = [['']]

    for i, dataset_ in enumerate(data_list):
        for j, entry in enumerate(dataset_):
            # print(entry)
            paragraph_text = entry["context"]
            question_text = entry["question"]
            start_index = entry['startidx']
            end_index = entry['endidx']
            if pred_list:
                prediction = pred_list[j]
            else:
                prediction = entry['pred']

            query_tokens = tokenizer.tokenize(question_text)
            lengths[i].append(len(query_tokens))
            if start_index != 0 or end_index != 0:
                answer_tokens = tokenizer.tokenize(prediction)
                if len(answer_tokens) != 0:
                    answer_lengths[i][entry['feedback']].append(len(answer_tokens))
                    answer_lengths[i]['All'].append(len(answer_tokens))

    bins = [i*2 for i in range(50)]
    plt.hist(answer_lengths[0]['All'], bins=bins)
    plt.xlabel('Lengths')
    plt.ylabel('# Examples')
    plt.title('Histogram of Answer Lengths')
    plt.xlim(0, 100)
    plt.grid(True)
    plt.savefig('Histogram_of_Answer_Length.png')

    for i, length in enumerate(lengths):
        print('Avg Question Length of Round %d: %f'%(i, sum(length)/len(length)))
        print('==================================')
        print('Avg Answer Length of Round %d'%i)
        report_strings = []
        answer_dict = answer_lengths[i]
        for k, v in answer_dict.items():
            report_strings.append("%s: %f"%(k, sum(v)/len(v)))
        print(' | '.join(report_strings))

        csv_data.append(['Question Length', '%2.2f'%(sum(length)/len(length))])
        csv_data.append([''])
        csv_data.append(['', 'Correct', 'Partially Correct', 'Wrong', 'All'])
        csv_data.append(['Answer Lengths'] + ['%2.2f'%(sum(v)/len(v)) for k, v in answer_dict.items()])
    return csv_data

def compute_question_type(data, round_idx):
    # for question type
    # What, When, Where, Why, How, How many, How much, 
    csv_data = [['']]
    re_list = [r'^[wW]hen|^[iIoO]n what year|^[aA]t what age|[wW]hat year|[wW]hat age|[wW]hat time', r'^[wW]ho', r'^[wW]here|In what city|In which city|In what country|In which country|[wW]hat city|[wW]hat country|[wW]hich city|[wW]hich country', r'^[wW]hy', r'^[wW]hich', r'^[wW]hat', r'^[hH]ow many|^[hH]ow much', r'^[hH]ow', r'^[dD]o|^[dD]oes|^[dD]id|[aA]re|[iI]s|[wW]as|[wW]ere|[cC]an|[cC]ould|[hH]a[sd]|[hH]ave']
    types = ['When', 'Who', 'Where', 'Why', 'Which', 'What', 'How many | How much', 'How (others)', 'Yes/No']
    data_list = [data]
    for i, dataset_ in enumerate(data_list):
        type_count = {'When':0, 'Who':0, 'Where':0, 'Why':0, 'Which':0, 'What':0, 'How many | How much':0, 'How (others)':0, 'Yes/No':0, 'Misc':0}
        for entry in dataset_:
            prediction = entry['pred']
            question_text = entry["question"]

            no_match = True
            for j, r in enumerate(re_list):
                if re.match(r, question_text) != None:
                    # print(re.match(r, question_text), question_text)
                    type_count[types[j]] += 1
                    no_match = False
                    break
            if no_match:
                type_count['Misc'] += 1
                print(question_text)

        print(type_count)
        total = sum([v for k, v in type_count.items()])
        csv_data.append([''] + [k for k, v in type_count.items()])
        csv_data.append(['Question Type Count'] + [v for k, v in type_count.items()])        
        csv_data.append(['Question Type (%)'] + ['%2.2f'%(100*v/float(total)) for k, v in type_count.items()])

    return csv_data


def compute_answer_lengths(data_list, pred_list, tokenizer):
    all_lengths = []
    answer_lengths = [{'Correct':[], 'Partially Correct':[], 'Wrong':[], 'All':[]} for _ in range(len(data_list))]
    csv_data = [['']]

    for i, dataset_ in enumerate(data_list):
        preds = pred_list[i]
        for index, entry in enumerate(dataset_):
            answer = preds[str(index)]
            if answer != "":
                answer_tokens = tokenizer.tokenize(answer)
                if len(answer_tokens) != 0:
                    # assert len(answer_tokens) == (end_index - start_index + i)
                    answer_lengths[i][entry['feedback']].append(len(answer_tokens))
                    answer_lengths[i]['All'].append(len(answer_tokens))
                    all_lengths.append(len(answer_tokens))

    for i, dataset_ in enumerate(data_list):
        print('==================================')
        print('Avg Answer Length of Round %d'%i)
        report_strings = []
        answer_dict = answer_lengths[i]
        for k, v in answer_dict.items():
            report_strings.append("%s: %f"%(k, sum(v)/len(v)))
        print(' | '.join(report_strings))

    print('Avg: %2.2f'%(sum(all_lengths)/float(len(all_lengths))))


if __name__ == '__main__':
    answer_source = 'short'
    round1_data = get_feedback_data('data/bandit_parallel/train/round1/train-data-parallel-round1-200-wprob-%s.jsonl.gz'%(answer_source))
    # round1_data = get_feedback_data('data/bandit/valid/Dev-400-full-annotations.jsonl.gz')
    tokenizer = DebertaV2TokenizerFast.from_pretrained('microsoft/deberta-v3-base', return_offsets_mapping=True)
    round_idx = 1

    csv_data = []
    csv_data += analyze_data(round1_data, round_idx)

    pred_list=None
    csv_data += compute_question_answer_lengths(round1_data, tokenizer, round_idx, pred_list=pred_list, is_training=True)
    csv_data += compute_question_type(round1_data, round_idx)
    print(csv_data[2])
    with open('out.tsv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerows(csv_data)

