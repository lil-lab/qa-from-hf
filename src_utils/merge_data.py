import gzip 
import json
import argparse

def get_data(input_file):
    with gzip.GzipFile(input_file, 'r') as reader:
        # skip header
        content = reader.read().decode('utf-8').strip().split('\n')
        input_data = [json.loads(line) for line in content]

    return input_data

def test_same(data_1, data_2):
    for d1, d2 in zip(data_1, data_2):
        assert d1['question'] == d2['question']

def merge(file_list, outfile):
    data = []
    for f_ in file_list:
        data += get_data(f_)

    fw = open(outfile + '.jsonl', 'w')
    for l in data:
        fw.write(json.dumps(l))
        fw.write('\n')
    fw.close()

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", default=None, type=str, required=True, help='data you wish to merge to (output)')
    parser.add_argument('-f','--input_file', action='append', help='<Required> Set flag', required=True)
    args = parser.parse_args()

    print(args.input_file)
    
    filename = args.outfile
    merge(args.input_file, filename)
