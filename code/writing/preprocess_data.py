import json
import argparse
import os
from collections import Counter


def read_data(filepath):
    data = []
    with open(filepath, 'r') as fin:
        for line in fin:
            single = json.loads(line)
            counters_gold_labels = Counter(single['labels'])
            if len(counters_gold_labels) != 1:  # ignore label of [false presupposition, normal] in training data
                continue
            elif list(counters_gold_labels.keys())[0] == 'false presupposition':
                data.append(single)
            else:  # normal
                continue
    return data


def extend_data(data):
    extended_data = []
    for single in data:
        presuppositions = single['presuppositions']
        corrections = single['corrections']

        assert len(presuppositions) == len(corrections) and len(presuppositions) != 0

        for pres, corr in zip(presuppositions, corrections):
            copied = single.copy()
            updated_pres = pres
            updated_corr = 'It is not the case that ' + pres[0].lower() + pres[1:]
            copied['presuppositions'] = [updated_pres]
            copied['corrections'] = [updated_corr]
            copied.pop("raw_labels")
            copied.pop("raw_presuppositions")
            copied.pop("raw_corrections")
            extended_data.append(copied)
    return extended_data


def flatten_data(data):
    flatten = []
    for single in data:
        presuppositions = single['presuppositions']
        corrections = single['corrections']

        assert len(presuppositions) == len(corrections) and len(presuppositions) != 0

        for pres, corr in zip(presuppositions, corrections):
            copied = single.copy()
            copied['presuppositions'] = [pres]
            copied['corrections'] = [corr]
            copied.pop("raw_labels")
            copied.pop("raw_presuppositions")
            copied.pop("raw_corrections")
            flatten.append(copied)
    return flatten


def write_data(output_path, data):
    with open(output_path, 'w+') as fout:
        for single in data:
            fout.write(json.dumps(single))
            fout.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, default='../../data/labeled_data/', help='data directory')
    parser.add_argument('--setting', choices=['dedicated', 'unified'], default='dedicated', help='dedicated: '
                                                                                                 'flatten the instance'
        'unified - additionally create instance of "It is not the case that" before presupposition for correction')
    parser.add_argument('--output-dir', type=str, required=True, help='output directory')
    args = parser.parse_args()

    splits = ['train', 'dev', 'test']

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    path = args.data_dir
    for split in splits:
        data_path = os.path.join(path, f'{split}.jsonl')
        data = read_data(data_path)

        if args.setting == 'dedicated':
            flatten = flatten_data(data)
        else:  # unified
            extended = extend_data(data)
            flatten = flatten_data(data)
            flatten.extend(extended)

        output_path = os.path.join(args.output_dir, f'{split}_{args.setting}.jsonl')
        write_data(output_path, flatten)


if __name__ == '__main__':
    main()
