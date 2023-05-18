import json
import csv
import argparse

from train_initial import get_data, byte_slice, read_squad_dataset
from IPython import embed
import pathlib



def main(args):
    

    pred_file = args.pred_file
    data_file = args.data_file

    if args.test_on_squad:
        eval_dataset = read_squad_dataset(data_file)
    else:
        eval_dataset = get_data(data_file)

    # reads it back
    with open(pred_file, "r") as f:
        preds_data = f.read()
    # decoding the JSON to dictionay
    preds = json.loads(preds_data)

    
    total = 0
    predicted_no_answer = 0
    accuracy = 0
    parsed = [['qid', 'question', 'annotation', 'prediction', 'context' if args.test_on_squad else 'document_url']] 

    for entry in eval_dataset:
        qid = entry['example_id']
        question = entry['question_text']
        if args.test_on_squad:
            orig_answer_text = '; '.join([l['orig_answer_text'] for l in entry['annotations']])
        else:
            orig_answer_text = byte_slice(text=entry['document_plaintext'],
                                          start=entry['annotations'][0]['minimal_answer']['plaintext_start_byte'],
                                          end=entry['annotations'][0]['minimal_answer']['plaintext_end_byte'])
        pred = preds[str(qid)]
        total += 1
        if len(pred) == 0:
            predicted_no_answer += 1
            if len(orig_answer_text) == 0:
                accuracy += 1
        else:
            if len(orig_answer_text) != 0:
                accuracy += 1

        parsed.append([qid, question, orig_answer_text, pred, entry['context'] if args.test_on_squad else entry['document_url']])
    
    print('total: %d | predicted_no_answer: %d \n no answer percentage: %2.3f \n No Answer Accuracy: %2.2f'%(total, predicted_no_answer, (predicted_no_answer/float(total))*100, (accuracy/float(total)*100)))


    pathlib.Path('/'.join(args.outfile.split('/')[:-1])).mkdir(parents=True, exist_ok=True)

    file = open(args.outfile, 'w', newline='')

    # writing the data into the file
    with file:
        write = csv.writer(file, delimiter='\t')
        write.writerows(parsed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", default=None, type=str, required=True)
    parser.add_argument("--data_file", default=None, type=str, required=True)
    parser.add_argument("--outfile", default=None, type=str, required=True, help='output file')
    parser.add_argument('--test_on_squad',
                        action='store_true',
                        help='whether test on SQuAD Dev data')
    args = parser.parse_args()

    main(args)