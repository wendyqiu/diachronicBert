# HistBERT: Diachronic BERT-based Word Embeddings on Historical Texts

## Introduction

HistBERT is the pretrained BERT (Bidirectional Encoder Representations from Transformers) embeddings on historical text data. Specifically, the language model is trained on the Corpus of Historical American English (COHA), from 1900s to 2000s.

## Outline

This document provides detailed guidelines on how to train HistBERT:

1. Preprocess COHA texts
2. Google TPU Setup
3. Continue BERT pretraining on historical data
4. Evaluation

## Data Preparation

Download historical text data from [COHA] (https://www.english-corpora.org/coha/). As a prototype, the current HistBERT only covers ten decades (from 1900s to 2000s). We selected

## Google TPU Setup

## Pretraining

```
python create_pretraining_data.py 
  --input_file=gs://historical_bert_2611/processed/1900.txt 
  --output_file=gs://historical_bert_2611/tmp/coach_1900.tfrecord 
  --vocab_file=gs://historical_bert_2611/uncased_L-12_H-768_A-12/vocab.txt 
  --do_lower_case=True 
  --max_seq_length=128  
  --max_predictions_per_seq=20 
  --masked_lm_prob=0.15 
  --random_seed=12345  
  --dupe_factor=5
```

## Evaluation
