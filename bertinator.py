'''
Simple application of distilbert
'''

from transformers import *
import torch


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')


def get_bert_vector(string):
    input_ids = torch.tensor(tokenizer.encode(f"[CLS] {string} [SEP]")).unsqueeze(0)
    outputs = model(input_ids)
    bert_vector = outputs[0][0, -1, :]
    
    return bert_vector
