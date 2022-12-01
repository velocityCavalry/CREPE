import os
import argparse
import numpy as np
import json
import csv
import sys

from collections import Counter

sys.path.append('../')
from evaluation import evaluate_detection


def get_question2label(file):
    data = []
    with open(file, 'r') as fin:
        for line in fin:
            instance = json.loads(line)
            question = instance['question']
            counters_gold_labels = Counter(instance['labels'])
            id = instance["id"]
            if len(counters_gold_labels) != 1:
                label = [1, 0]
            else:
                if list(counters_gold_labels.keys())[0] == 'false presupposition':
                    label = [1, 1]  # false presupposition label to 1
                else:
                    label = [0, 0]  # normal label to 1
            data.append((id, question, label))

        data = sorted(data, key=lambda x: x[0])  # sort based on id
        questions = [x[1] for x in data]
        labels = [x[2] for x in data]

        return questions, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-file', type=str, default='../../data/labeled_data/test.jsonl')

    np.random.seed(2022)

    args = parser.parse_args()
    dev_question_path = args.test_file

    dev_questions, groundtruths = get_question2label(dev_question_path)
    always_fp = [1 for _ in range(len(groundtruths))]
    always_normal = [0 for _ in range(len(groundtruths))]
    random = [int(np.random.choice([0, 1])) for _ in range(len(groundtruths))]

    permissive_always_fp_f1 = evaluate_detection(always_fp, groundtruths)
    permissive_always_normal_f1 = evaluate_detection(always_normal, groundtruths)
    permissive_random_f1 = evaluate_detection(random, groundtruths)

    print("always fp macro-f1", permissive_always_fp_f1)
    print("always normal macro-f1", permissive_always_normal_f1)
    print("random macro-f1", permissive_random_f1)


if __name__ == '__main__':
    main()