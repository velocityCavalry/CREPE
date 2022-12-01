# Baseline Code

1.[Detection Subtask](#detection-subtask)
   * [Trivial Baselines](#trivial-baselines)
   * [Gold-Comment Track](#detection-gold-comment-track)
   * [Main Track](#detection-main-track)
2. [Writing Subtask](#writing-subtask)
   * [Preprocessing](#preprocessing)
   * [Copy Baselines](#copy-baseline)
   * [Gold-Comment Track](#writing-gold-comment-track)
   * [Main Track](#writing-main-track)


## Detection Subtask
We provide 4 trivial baselines, 5 Gold-Comment track baselines, and 2 Main track baselines. 

### Trivial Baselines

To run our nearest neighbor baselines, you need to install FAISS first
```shell
# CPU-only version
$ conda install -c pytorch faiss-cpu

# GPU(+CPU) version
$ conda install -c pytorch faiss-gpu

# or for a specific CUDA version
$ conda install -c pytorch faiss-gpu cudatoolkit=10.2 # for CUDA 10.2
```

For a detailed instruction of how to install FAISS and FAQs, please refer to the [FAISS installation guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) 
in their repository.

We provide `trivial_baselines/requirement.txt` for packages that is necessary to install. As a reference, we also provide the *full* environment that we used to run our trivial baselines at `trivial_baselines/environment.yml`.

To run our baselines:

```shell

# Random selection baseline, always predict FP baseline, always predict normal baseline
python code/trivial_baselines/always_one_choice.py --test-file {either test.jsonl or dev.jsonl}
```

To download the embedding resources (5.4 MB) and the index: [google drive](https://drive.google.com/file/d/1jpKJsQUQqg3QMtD0ufPfHE9BoDfmWnfO/view?usp=sharing) 
or run `data/download_resources.sh`.

The directory contains:
* `dev_question_embs_L=64.npy`: c-REALM embedding for the validation questions
* `test_question_embs_L=64.npy`: c-REALM embedding for the test questions
* `train_question_embs_L=64.npy`: c-REALM embedding for the train questions
* `train_question.IndexFlatIP`: FAISS index, can also be obtained

After downloading the resources, run our nearest neighbor search using the following command:
```shell
# Nearest Neighbor baselines
python code/trivial_baselines/nearest_neighbor.py --index {path to train question index} \
    --dev-file {path to dev or test file}  --dev-emb {path to dev or test embedding} \ 
    --train-file {path to train file} -train-emb {path to train embedding}
```

### Detection Gold-Comment Track
TBA

### Detection Main Track 
TBA

## Writing Subtask

### Preprocessing
We provide a preprocessing script for our writing task at `code/writing/preprocess_data.py`. 
You can run it by 
```shell
python preprocess_data.py --data-dir {the directory of your data} \
                          --setting {dedicated, unified} \
                          --output-dir {output directory of the processed file}
```

The `dedicated` setting will flatten each `false presupposition` instance to contain only one pair of presupposition and correction.
The `unified` setting will flatten and extend each `false presupposition` instance to include the prefix `It is not the case that + presupposition` as the correction.  
```text
# original 
{'presuppositions': [presupposition 1, presupposition 2],
'corrections': [correction 1, correction 2]}

# dedicated (flattened)
{'presuppositions': [presupposition 1], 'corrections': [correction 1]}
{'presuppositions': [presupposition 2],'corrections': [correction 2]}

# unified
{'presuppositions': [presupposition 1], 'corrections': [correction 1]}
{'presuppositions': [presupposition 2],'corrections': [correction 2]}
{'presuppositions': [presupposition 1], 'corrections': ['It is not the case that ' + presupposition 1]}
{'presuppositions': [presupposition 2],'corrections': ['It is not the case that ' + presupposition 2]}
```
Each line of the output files contain a dictionary of the format

```python
{
 'id': id, 
 'question': question, 
 'comment': comment, 
 'passages': [passage 1, passage 2, ..., passage 25], 
 'labels': ["false presuppositions"], 
 'presuppositions': [presupposition],
 'corrections': [correction]
 }
```

### Copy Baseline
TBA

### Writing Gold-Comment Track 
TBA

### Writing Main Track
TBA
