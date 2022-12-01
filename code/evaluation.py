import argparse
import csv
import json
import numpy as np
import re
import string
from collections import Counter
from datasets import load_metric


def read_file(path):
    id2data = []
    with open(path, 'r') as fin:
        for line in fin:
            curr_obj = line.strip()
            assert 'id' in curr_obj, \
                "please include question id in the file to avoid prediction and groundtruth order mismatch"
            id2data[curr_obj['id']] = curr_obj['prediction']
    return id2data


def read_reference_file(path, setting='detection'):
    data = []
    with open(path, 'r') as fin:
        for line in fin:
            instance = json.loads(line)
            counters_gold_labels = Counter(instance['labels'])
            if setting == 'detection':
                if len(counters_gold_labels) != 1:
                    raise ValueError("label is not fp or normal", counters_gold_labels)
                else:
                    if list(counters_gold_labels.keys())[0] == 'false presupposition':
                        label = [1, 1]  # false presupposition label to 1
                    else:
                        label = [0, 0]  # normal label to 1
                data.append(label)
            elif setting.startswith('writing'):
                if len(counters_gold_labels) != 1:
                    continue
                else:
                    if list(counters_gold_labels.keys())[0] != 'false presupposition':
                        continue
                if setting.endswith('correction'):
                    data.append(instance['corrections'])
                if setting.endswith('presupposition'):
                    data.append(instance['presuppositions'])
    return data


def read_text_file(path):
    data = []
    with open(path, 'r') as fin:
        for line in fin:
            normalized = line.strip()
            data.append(normalized)
    return data


def fill_groudtruth(data):
    groundtruths = []
    max_len = max([len(x) for x in data])

    for gt in data:
        if len(gt) != max_len:
            for _ in range(max_len - len(gt)):
                gt.append(gt[0])
        groundtruths.append(gt)
    assert [len(x) == max_len for x in groundtruths]
    return groundtruths


def normalize(predictions):

    assert (type(predictions[0]) == int or type(predictions[0]) == str or type(predictions[0]) == list)

    normalized_predictions = []
    for prediction in predictions:
        if type(prediction) == int:
            if prediction != 0 and prediction != 1:
                raise ValueError("unknown prediction", prediction)
            else:
                normalized_predictions.append(prediction)
        elif type(prediction) == str:
            if prediction == 'false presupposition':
                normalized_predictions.append(1)
            elif prediction == 'normal':
                normalized_predictions.append(0)
            else:
                raise ValueError("unknown prediction", prediction)
        else:  # type(prediction) == list:
            if prediction == [1, 1]:
                normalized_predictions.append(1)
            elif prediction == [0, 0]:
                normalized_predictions.append(0)
            else:
                raise ValueError("unknown prediction", prediction)

    return normalized_predictions


def read_writing_predictions(path, setting='writing-presupposition'):
    predictions = []
    with open(path, 'r') as fin:
        for line in fin:
            curr_instance = json.loads(line)
            if setting == 'writing-presupposition':
                assert 'presupposition' in curr_instance
                predictions.append(curr_instance['presupposition'])
            if setting == 'writing-correction':
                assert 'correction' in curr_instance
                predictions.append(curr_instance['correction'])
    return predictions


def evaluate_detection(predictions, groundtruths):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

    for gold_labels, prediction in zip(groundtruths, predictions):
        assert type(gold_labels) == list and type(prediction) == int
        is_correct = prediction == 1 if (1 in gold_labels) else prediction == gold_labels[0]

        if prediction == 1 and is_correct:
            true_pos += 1
        elif prediction == 0 and is_correct:
            true_neg += 1
        elif prediction == 1 and not is_correct:
            false_pos += 1
        elif prediction == 0 and not is_correct:
            false_neg += 1

    if true_pos == 0:
        f1_fp = 0.0  # in this case, f1 is always 0
    else:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1_fp = 2 * precision * recall / (precision + recall)

    if false_neg == 0:
        f1_normal = 0.0
    else:
        precision = true_neg / (true_neg + false_neg)
        recall = true_neg / (true_neg + false_pos)
        f1_normal = 2 * precision * recall / (precision + recall)

    return (f1_fp + f1_normal) / 2.0


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# F1 score definition
def local_f1_score(prediction, groundtruths):
    max_f1 = 0
    for ground_truth in groundtruths:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        max_f1 = max(f1, max_f1)
    return max_f1


def evaluate_writing(groundtruths, predictions):
    bleu = load_metric('sacrebleu')
    for gt in groundtruths:
        assert len(gt) == 6
    bleu_result = bleu.compute(predictions=predictions, references=groundtruths)
    print(' -- res --: bleu', round(bleu_result["score"], 2))

    total_f1 = 0
    total_len = 0
    for idx, prediction in enumerate(predictions):
        unigram_f1 = local_f1_score(prediction, groundtruths[idx])
        total_f1 += unigram_f1
        total_len += 1
    total_f1 = total_f1 / total_len
    print(' -- res --: f1', total_f1)
    return round(bleu_result["score"], 2), total_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-path', type=str, required=True, help='path to the original dev .jsonl file'
                                                                          'or original test .jsonl file')
    parser.add_argument('--prediction-path', type=str, required=True, help='path to the prediction, '
                                                                           'for detection subtask, '
                                                                           'it can be a .jsonl file with predictions '
                                                                           'in field "prediction" as either integer '
                                                                           '(0, 1) or string (false presupposition '
                                                                           'or normal), or in .npy file that contains '
                                                                           'the raw logits from our model, '
                                                                           'for writing task, it can be a .txt '
                                                                           'or a .jsonl file')
    parser.add_argument('--subtask', choices=['detection', 'writing-presupposition', 'writing-correction'], required=True)
    args = parser.parse_args()

    assert args.reference_path.endswith('.jsonl')
    references = read_reference_file(args.reference_path, args.subtask)

    if args.subtask == 'detection':
        if args.prediction_path.endswith('.jsonl'):
            predictions = read_file(args.prediction_path)
            assert len(predictions) == len(references), "reference and prediction not the same length"
            normalized_predictions = normalize(predictions)
            macro_f1 = evaluate_detection(normalized_predictions, references)
        elif args.prediction_path.endswith('.npy'):
            prediction_logits = np.load(args.prediction_path)
            assert prediction_logits.shape == (len(references), 2)
            predictions = np.argmax(prediction_logits, axis=1).tolist()
            macro_f1 = evaluate_detection(predictions, references)
        else:
            raise ValueError("Not supported file format. Please use .jsonl or .npy for detection task now")

        print("the macro-f1 is", macro_f1)
    elif args.subtask == 'writing-presupposition' or args.subtask == 'writing-correction':
        normalized_references = fill_groudtruth(references)
        if args.prediction_path.endswith('.txt'):
            predictions = read_text_file(args.prediction_path)
        elif args.prediction_path.endswith('.jsonl'):
            predictions = read_writing_predictions(args.prediction_path, setting=args.subtask)
        else:
            raise ValueError("Not supported file format. Please use .jsonl or .txt for writing task now")
        evaluate_writing(normalized_references, predictions)


if __name__ == '__main__':
    main()
