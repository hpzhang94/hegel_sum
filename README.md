# HEGEL: Hpyergraph Transformer for Long Document Summarization
source code for EMNLP 2022 paper [HEGEL: Hypergraph Transformer for Long Document Summarization](https://arxiv.org/abs/2210.04126)


## Directory Structure

* ./preprocess: save the document preprocessing scripts
* ./model_param: save the model checkpoints
* ./data: save the original and preprocessed data

## Preprocessing

```python
# extract topics
python ./preprocess/topic_model.py --data_path ./data

# extract keywords
python ./preprocess/keywords.py --data_path ./data
```

## Train & Test

### Train

```python
# train on ArXiv
python train.py --data_path [data_path] --data_name arxiv --hidden 4096  --model_save_path [checkpoint_path]  --epoch 25 --lr 1e-4 --dropout 0.3 --batch_size 32 --use_kw --use_lda --use_sec

# train on PubMed
python train.py --data_path [data_path] --data_name pubmed --hidden 4096 --model_save_path [checkpoint_path]  --epoch 25 --lr 1e-4 --dropout 0.3 --batch_size 32 --use_kw --use_lda --use_sec

```

### Test

```python
# test on ArXiv
python test.py --data_path [data_path] --data_name arxiv --hidden 4096  --batch_size 32  --use_kw --use_lda --use_sec --weight_path [checkpoint_path]

# test on PubMed
python test.py --data_path [data_path] --data_name pubmed --hidden 4096  --batch_size 32  --use_kw --use_lda --use_sec --weight_path [checkpoint_path]
```


## Downloads

* Our preprocessed data

  * ArXiv
  * PubMed
* The raw data:

  * ArXiv
  * PubMed
