import os
import argparse
import numpy as np
import faiss
import json
import csv
from collections import Counter

import sys
sys.path.append('../')
from evaluation import evaluate_detection


def get_question2label(file):
    data = []
    with open(file, 'r') as fin:
        for line in fin:
            instance = json.loads(line)
            question = instance['question']
            counters_gold_labels = Counter(instance['gold_labels'])
            id = instance["id"]
            if len(counters_gold_labels) != 1:
                label = [1, 0]
            else:
                if list(counters_gold_labels.keys())[0] == 'false presupposition':
                    label = [1, 1]
                else:
                    label = [0, 0]
            data.append((id, question, label))

        data = sorted(data, key=lambda x: x[0])  # sort based on id
        questions = [x[1] for x in data]
        labels = [x[2] for x in data]

        return questions, labels


def run_faiss_and_predict(index_path,
                         train_vec_path,
                         dev_vec_path,
                         train_questions,
                         dev_questions,
                         train_labels):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        train_vec = np.load(train_vec_path)
        print("shape of train vec:", train_vec.shape)  # (3462, 128)
        # Exact Search for Inner Product
        index = faiss.IndexFlatIP(train_vec.shape[1])
        index.add(train_vec)
        faiss.write_index(index, index_path)

    dev_vec = np.load(dev_vec_path)
    print("dev_vec:", dev_vec.shape)  # (2000, 128)
    D, I = index.search(dev_vec, 1)  # scores, indexes
    assert D.shape == I.shape == (dev_vec.shape[0], 1)  # number of questions x k
    prediction_indices = I.tolist()

    dev2train = []  # list might be easier to sample
    predictions = []

    for i in range(len(prediction_indices)):
        current_dev_question = dev_questions[i]
        neighbor_questions = []
        # each number corresponds to the idx in train question
        assert len(prediction_indices[i]) == 1
        for question_id in prediction_indices[i]:
            neighbor_questions.append(train_questions[question_id])
            if train_labels[question_id][0] == train_labels[question_id][1]:
                predictions.append(train_labels[question_id][0])
            else:
                predictions.append(int(np.random.choice(train_labels[question_id])))
        dev2train.append((current_dev_question, neighbor_questions))

    return dev2train, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str,
                        default='resources/train_question.IndexFlatIP')
    parser.add_argument('--dev-file', type=str, default='data/labeled_data/test.jsonl')
    parser.add_argument('--dev-emb', type=str, default='resources/test_question_embs_L=64.npy')
    parser.add_argument('--train-file', type=str, default='data/labeled_data/train.jsonl')
    parser.add_argument('--train-emb', type=str, default='resources/train_question_embs_L=64.npy')

    args = parser.parse_args()
    # assert that arguments are passed correctly
    assert 'train_question.IndexFlatIP' in args.index
    assert ('test' in args.dev_emb and 'test' in args.dev_file) or ('dev' in args.dev_emb and 'dev' in args.dev_file)
    assert 'train' in args.train_file and 'train' in args.train_emb

    np.random.seed(2022)

    train_questions, train_labels = get_question2label(args.train_file)
    dev_questions, groundtruths = get_question2label(args.dev_file)

    dev2train, predictions = run_faiss_and_predict(args.index, args.train_emb, args.dev_emb, train_questions,
                                                   dev_questions, train_labels)

    macro_f1 = evaluate_detection(predictions, groundtruths)

    print("macro f1", macro_f1)


if __name__ == '__main__':
    main()
