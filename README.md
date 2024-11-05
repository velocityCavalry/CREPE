# CREPE: Open-Domain Question Answering with False Presuppositions
This repository contains the data and code for the paper: [CREPE: Open-Domain Question Answering with False Presuppositions](https://arxiv.org/abs/2211.17257)
by [Xinyan Velocity Yu](https://velocitycavalry.github.io), [Sewon Min](https://shmsw25.github.io/), 
[Luke Zettlemoyer](https://www.cs.washington.edu/people/faculty/lsz) and [Hannaneh Hajishirzi](https://homes.cs.washington.edu/~hannaneh/).

## Content
1. [Download Data](#download-data)
2. [Dataset Contents](#data-contents)
    * [Statistics](#statistics)
    * [Format](#data-format)
3. [Evaluation Script](#evaluation-script)
4. [Citation](#citation)
5. [Contact](#contact)

Baseline code to be added :)

## Download Data

We host our data on google drive. You can either download the data directly [from google drive](https://drive.google.com/drive/folders/0BxawpCgUevCzfkctQUY4RXI5MzRDNk5YX2FHeVZJNU12NUZCbXVHT3FxQjluSXdGblNwbHM?resourcekey=0-iaePOJrAI_kPGjSok2ITZg&usp=sharing) 
or run the following command.

```shell
# download all the labeled data and unzip (42.2MB)
bash download_raw_data.sh

# download all files (labeled + unlabeled) and unzip (2.56GB)
bash download_raw_data.sh -s all

# download unlabeled training data only (2.52GB)
bash download_raw_data.sh -s unlabeled
```

## Data Contents

The data is built on top of the [ELI5](https://github.com/facebookresearch/ELI5) data. The train, development, and test set are split based on the time of the posting.

### Statistics

| Data Split      | # Questions | # Question w/ FP | FP % | Time Frame     |
|-----------------|-------------|------------------|------|----------------|
| Train           | 3,462       | 907              | 26.2 | 2011 - 2018    |
| Development     | 2,000       | 544              | 27.2 | Jan - Jun 2019 |
| Test            | 3,004       | 751              | 25.0 | Jul - Dec 2019 |
| Train Unlabled  | 196,385     | -                | -    | 2011 - 2018    |
| Total (Labeled) | 8,446       | 2,202            | 26.0 | -              | 
| Total (Labeled + Unlabeled) | 204, 851    | -    | -    | -              | 

### Data Format

Each line of `{train, dev, test}.jsonl` contains a dictionary that represents a single question, with the following keys:

* `id`(string): an identification of the question
* `question` (string): the question
* `comment` (string): the comment
* `labels` (list of strings): either `[normal]`, `[false_presupposition]`, or `[false_presupposition, normal]`. The last option occurs when there was an annotator disagreement, only in the training data.
* `presuppositions`(list of strings): a list of identified false presupposition. It is an empty list if `labels` is `normal`.
* `corrections`(list of strings): a list of identified corrections. It is an empty list if `labels` is `normal`.
* `passages` (list of strings): 25 passages from the English Wikipedia, retrieved by c-REALM (see the paper for details).

As additional resources (to investigate the ambiguity of the data), we provide the data written by the generators, before being validated by the validators.
following fields:

* `raw_labels` (list of strings): can contain both `normal` and `false_presupposition` 
* `raw_presuppositions` (list of strings): if `false_presupposition` is in `raw_labels`, `[]` otherwise
* `raw_corrections` (list of strings): if `false_presupposition` is in `raw_labels`, `[]` otherwise

For unlabeled training data, `labels`, `presuppositions`, or `corrections` are not included, but all others are included.

## Evaluation Script
Evaluation script can be found in `code/evaluation.py`. 

First, install the dependencies
```bash
pip install datasets==2.3.2 # any version between 2.3.2 and 2.7.1 works
pip install sacrebleu==2.1.0 # any version between 2.1.0 and 2.3.1 works
```

Then, format your prediction files.

* For the detection subtask, we support the following two formats.
    * `.jsonl` file with each line formatted like `{'prediction': <your prediction>}`, with `<your prediction>` 
being either an integer `0 or 1, 1 stands for false presupposition`, or a string `"false presupposition" or "normal"`.
    * `.npy` file containing the logits of shape `(len(references), 2)`, and we will obtain the prediction by taking an `argmax` for
axis `1`. 
* For the writing subtask, we support the following two formats. (Note that you need two files, each for presuppositions and corrections).
    * `.txt` file that each line is a prediction for one false presupposition instance.
    * `.jsonl` file: if you are evaluating for presuppositions, it should contain `{'presupposition': <your presupposition>}`,
or if you are evaluating for corrections, it should contain `{'correction': <your correction>}` for each line. `<your presupposition>` should be a string.

Finally, run the following command.
```shell
cd code
python evaluation.py --reference-path {the reference path to the original dev or test file} \
                     --prediction-path {the path to the prediction file, following the above format} \
                     --subtask {'detection', 'writing-presupposition', 'writing-correction'}
```

We provide 3 `dev` sample output files in the `sample-outputs/`. See the paper for more details of each system.
* `detection-gold-comment-track-combined.npy`: the `numpy` logit for the Question+Comment model for gold-comment track `detection` task.
* `writing-gold-comment-track-unified-correction-predictions.txt`: the correction predictions for the unified model for gold-comment track `writing-correction` task.
* `writing-gold-comment-track-unified-presupposition-predictions.txt`: the presupposition predictions for the unified model for gold-comment track `writing-presupposition` task.

For a quick start, verifying the results by running 
```shell
python evaluation.py --reference-path ../data/labeled_data/dev.jsonl \
                     --prediction-path ../sample-outputs/writing-gold-comment-track-unified-presupposition-predictions.txt \
                     --subtask writing-presupposition
```

## Citation
If you find the CREPE dataset or false presupposition detection dataset useful, please cite our paper:
```
@article{yu2022crepe,
    title={CREPE: Open-Domain Question Answering with False Presuppositions}, 
    author={Xinyan Velocity Yu and Sewon Min and Luke Zettlemoyer and Hannaneh Hajishirzi},
    year={2022},
    journal={arXiv preprint arXiv:2211.17257},
    url={https://arxiv.org/abs/2211.17257}
}
```

Please also make sure to credit and cite the creators of ELI5 dataset, the dataset that we build ours upon:

```
@inproceedings{fan2019eli5,
    title = "{ELI}5: Long Form Question Answering",
    author = "Fan, Angela and Jernite, Yacine and Perez, Ethan and Grangier, David and Weston, Jason and Auli, Michael",
    booktitle = "Proceedings of the Annual Meeting of the Association for Computational Linguistics",
    year = "2019",
}
```

## Contact
If you have any question, please email `{xyu530, sewon}` [at] `cs.washington.edu`. Thanks!
