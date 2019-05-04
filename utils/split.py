#!/usr/bin/env python3
import argparse
import random
import os
'''
usage: python3 utils/dedup_and_split.py file --train 0.5 --valid 0.2 --test 0.3
'''

SEP = ' # '

def parse_and_dedup_examples(filename, sep=SEP):
    '''
    It is assumed that each line in the file constitutes one example, and if the
    separator = #, the lines are of the form of strings 'label#example'.
    '''
    labels = []
    examples = []
    with open(filename, 'r') as f:
        examples_lines = f.read().splitlines()
    examples_lines = [e for e in examples_lines if e]  # remove empty lines
    examples_lines = list(set(examples_lines))  # remove duplicates
    random.shuffle(examples_lines)
    for e in examples_lines:
        e_sep = e.split(sep)
        labels.append(e_sep[0])
        examples.append(e_sep[1])
    return labels, examples


def partition_list(l, partition_sizes):
    assert sum(partition_sizes) == 1.
    partitioned = []
    length = len(l)
    for s in partition_sizes:
        if l:
            index = round(s * length)
            partitioned.append(l[:index])
            l = l[index:]
    return partitioned


def split_to_train_valid_test(labels, examples, train=.6, valid=.3, test=.1):
    assert train + valid + test == 1.
    classes_examples = {c: [] for c in set(labels)}
    for i in range(len(labels)):
        classes_examples[labels[i]].append(examples[i])
    train_labels, train_examples = [], []
    valid_labels, valid_examples = [], []
    test_labels, test_examples = [], []
    for c in classes_examples:
        train_part, valid_part, test_part = \
            partition_list(classes_examples[c], (train, valid, test))
        train_examples.extend(train_part)
        valid_examples.extend(valid_part)
        test_examples.extend(test_part)
        train_labels.extend([c for i in train_part])
        valid_labels.extend([c for i in valid_part])
        test_labels.extend([c for i in test_part])
    return (train_labels, valid_labels, test_labels), \
           (train_examples, valid_examples, test_examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split file with training examples into three balanced.")
    parser.add_argument(
        'filename',
        type=str,
        help="Path to file with examples.")
    parser.add_argument('--train', type=float, default=0.6, help="Train set size.")
    parser.add_argument('--valid', type=float, default=0.1, help="Validation set size.")
    parser.add_argument('--test', type=float, default=0.3, help="Test set size.")
    args = parser.parse_args()
    assert args.train + args.valid + args.test == 1.

dirname = os.path.dirname(args.filename)
labels, examples = parse_and_dedup_examples(args.filename)
(train_labels, valid_labels, test_labels), \
    (train_examples, valid_examples, test_examples) = \
    split_to_train_valid_test(labels, examples, args.train, args.valid, args.test)

with open(os.path.join(dirname, 'train.expr'), 'w') as f:
    f.write('\n'.join(train_examples) + '\n')
with open(os.path.join(dirname, 'train.rmd'), 'w') as f:
    f.write('\n'.join(train_labels) + '\n')

with open(os.path.join(dirname, 'valid.expr'), 'w') as f:
    f.write('\n'.join(valid_examples) + '\n')
with open(os.path.join(dirname, 'valid.rmd'), 'w') as f:
    f.write('\n'.join(valid_labels) + '\n')

with open(os.path.join(dirname, 'test.expr'), 'w') as f:
    f.write('\n'.join(test_examples) + '\n')
with open(os.path.join(dirname, 'test.rmd'), 'w') as f:
    f.write('\n'.join(test_labels) + '\n')

with open(os.path.join(dirname, 'vocab.expr'), 'w') as f:
    vocab = set(' '.join(examples).split(' '))
    f.write('\n'.join(vocab) + '\n')
with open(os.path.join(dirname, 'vocab.rmd'), 'w') as f:
    vocab = set(' '.join(labels).split(' '))
    f.write('\n'.join(vocab) + '\n')
