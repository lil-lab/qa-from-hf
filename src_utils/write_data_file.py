from pathlib import Path
import argparse

def write_train(args):
    with open(args.train_file, 'w') as fw:
        if args.r_idx == 1:
            fw.write((args.train_path + ('train-data-round1.jsonl.gz')))
        elif args.r_idx == 2:
            fw.write((args.train_path  + ('train-data-round2.jsonl.gz\n')))
            fw.write((args.train_path  + ('train-data-round1.jsonl.gz')))
        else:
            fw.write((args.train_path  + ('train-data-round%d.jsonl.gz\n'%(args.r_idx))))
            fw.write((args.train_path  + ('train-data-round1to%d.jsonl.gz\n'%(args.r_idx-1))))

def write_train_parallel(args):
    with open(args.train_file, 'w') as fw: 
        if args.r_idx == 1:
            fw.write((args.train_parallel_path + ('round1/') + ('train-data-parallel-round1-%s.jsonl.gz'%(args.variant))))
        elif args.r_idx == 2:
            fw.write((args.train_parallel_path + ('round2/') + ('train-data-parallel-round2-%s.jsonl.gz\n'%(args.variant))))
            fw.write((args.train_parallel_path + ('round1/') + ('train-data-parallel-round1-%s.jsonl.gz'%(args.variant))))
        else:
            fw.write((args.train_parallel_path + ('round%d/'%args.r_idx) + ('train-data-parallel-round%d-%s.jsonl.gz\n'%(args.r_idx, args.variant))))
            fw.write((args.train_parallel_path + ('round%d/'%args.r_idx) + ('train-data-parallel-round1to%d-%s.jsonl.gz\n'%(args.r_idx-1, args.variant))))

if __name__ == '__main__':
    """
        Examples:
        for long-term:       python src_utils/write_data_file.py --exp long-term --r_idx 2
        for model variants:  python src_utils/write_data_file.py --exp variants  --r_idx 4 --variant fewer

        This script will write results to train_files.txt.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/train/')
    parser.add_argument('--train_parallel_path', type=str, default='data/train_parallel/')
    parser.add_argument('--exp', choices=['long-term', 'variants'], default='long-term')
    parser.add_argument('--r_idx', type=int, default=1)
    parser.add_argument('--train_file', type=str, default='train_files.txt')
    parser.add_argument('--variant', choices=['default', 'fewer', 'newsqa', 'noclass', 'weaker'], default='default')
    args = parser.parse_args()

    train_path = Path(args.train_path)
    train_parallel_path = Path(args.train_parallel_path)

    train_path.mkdir(parents=True, exist_ok=True)
    train_parallel_path.mkdir(parents=True, exist_ok=True)

    if args.exp == 'long-term':
        write_train(args)
    elif args.exp == 'variants':
        write_train_parallel(args)
    else:
        raise NotImplementedError  