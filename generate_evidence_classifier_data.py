"""
Uses 'chunks_to_foods.json' (from 'parse_json_labels.py') to create
a dataset of bertified vectors of the chunks and a corrospending
boolean label
"""
from transformers import *
import torch
import json
import pandas as pd
import numpy as np
from tqdm import tqdm


chunks_path = 'chunks_to_foods.json'

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')


if __name__ == "__main__":
    with open(chunks_path, 'r') as f:
        data = json.load(f)

    bert_vectors = []
    labels = []

    for chunk_item in tqdm(data):
        chunk = chunk_item["chunk"]
        foods = chunk_item["foods"]

        label = len(foods) > 0

        input_ids = torch.tensor(tokenizer.encode(f"[CLS] {chunk} [SEP]")).unsqueeze(0)
        outputs = model(input_ids)
        bert_vector = outputs[0][0, -1, :]

        bert_vectors += [bert_vector.detach().numpy()]
        labels += [label]

    bert_vectors_vector = np.array(bert_vectors)

    df_bert = pd.DataFrame(bert_vectors_vector, columns=[f'b{x}' for x in range(bert_vectors_vector.shape[1])])
    df_label = pd.DataFrame({'labels': labels})

    df = df_label.join(df_bert)

    print('saving file')
    df.to_csv('bert_data.csv')
